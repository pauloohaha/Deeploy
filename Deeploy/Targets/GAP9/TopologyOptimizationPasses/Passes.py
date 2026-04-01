# SPDX-FileCopyrightText: 2025 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0

import math
from functools import partial

import numpy as np
import onnx_graphsurgeon as gs

from Deeploy.CommonExtensions.OptimizationPasses.Matchers import Match, NonBranchingMatcher
from Deeploy.CommonExtensions.OptimizationPasses.PassClasses import ReplaceSequentialPatternPass, contextagnostic


def _ne16_conv_1x1_weight_layout(W, w_bits=8):
    """Pack int8 weights for NE16 1x1 conv mode.
    W: int8 [Ko, Ki] -> uint8 [Ko, Nb_KI, Qw, 2] (bitplane packed)
    Weights stored as uint8 = int8 + 128.
    """
    tp_in = 16
    Ko_, Ki_ = W.shape
    W_uint8 = (W.astype(np.int32) + 128).astype(np.uint8)
    nb_ki = (Ki_ + tp_in - 1) // tp_in
    w_binary = np.zeros((Ko_ * nb_ki, w_bits, 8, tp_in // 8), dtype=np.uint8)
    for ko in range(Ko_):
        for ki_maj in range(nb_ki):
            for ki_min in range(tp_in):
                idx = ko * nb_ki + ki_maj
                ki = ki_maj * tp_in + ki_min
                val = int(W_uint8[ko, ki]) if ki < Ki_ else 0
                for q in range(w_bits):
                    w_binary[idx, q, ki_min % 8, ki_min // 8] = (val >> q) & 1
    space = np.logspace(0, 7, num=8, base=2, dtype=np.int32).reshape((8, 1))
    w_layout = np.sum(w_binary * space, axis=2, dtype=np.uint8)
    return w_layout.reshape((Ko_, nb_ki, w_bits, tp_in // 8))


def _compute_ne16_scale_shift(mul_values, log2D):
    """Convert Deeploy's mul/log2D to NE16's per-channel scale/scale_n."""
    Ko = len(mul_values)
    ne16_scale = np.zeros(Ko, dtype=np.uint8)
    ne16_scale_n = np.zeros(Ko, dtype=np.uint8)
    for ko in range(Ko):
        sf = float(mul_values[ko]) / float(2 ** log2D)
        if sf >= 1.0:
            sn = 0
            sc = min(255, max(1, int(round(sf))))
        elif sf > 0:
            sn = min(31, max(0, int(math.floor(math.log2(127.0 / sf)))))
            sc = min(255, max(1, int(round(sf * (1 << sn)))))
        else:
            sn = 0
            sc = 0
        ne16_scale[ko] = sc
        ne16_scale_n[ko] = sn
    return ne16_scale, ne16_scale_n


def _is_input_signed(node, graph):
    """Check if the data input to this GEMM node is signed by tracing the ONNX graph."""
    input_tensor = node.inputs[0]
    for n in graph.nodes:
        if input_tensor in n.outputs:
            if n.op == 'Quant' and 'signed' in n.attrs:
                return bool(n.attrs['signed'] == 1)
            elif  n.op == "RequantizedGemm" and 'signed' in n.attrs:
                return bool(n.attrs['signed'].values == 1)
    return True  # default to signed


def _ne16_adjust_gemm_weight_layout_fun(graph: gs.Graph, match: Match, name: str):
    """Reformat GEMM weights into NE16 1x1 conv bitplane layout.

    Converts weight tensor from [Ko, Ki] int8 to [Ko, Nb_KI, Qw, 2] uint8 bitplane packed.
    This must run BEFORE tiling so the tiling system sees the packed shape.
    """
    matched_nodes = list(match.nodes_map.values())
    node = matched_nodes[0]

    # Weight is input[1] for both Gemm and RequantizedGemm
    weightTensor = node.inputs[1]

    if not isinstance(weightTensor, gs.Constant):
        return graph

    values = weightTensor.values

    # Skip true float weights (Deeploy stores int8 weights as float32)
    if not np.array_equal(values, np.round(values)):
        return graph

    # Check shape is 2D
    if len(values.shape) != 2:
        return graph

    # Determine actual Ko, Ki based on transB
    transB = node.attrs.get('transB', 0)
    if transB:
        Ko, Ki = values.shape
    else:
        Ki, Ko = values.shape

    # Check NE16 compatibility BEFORE modifying the node
    if Ki % 16 != 0:
        return graph

    # Safe to transpose — node will be fully processed
    if not transB:
        values = values.T  # local copy, now [Ko, Ki]
        node.attrs['transB'] = 1

    # Compute per-channel weight sum BEFORE packing (needed for signed input bias compensation)
    w_int8 = values.astype(np.int8)
    w_sum = w_int8.astype(np.int64).sum(axis=1)  # [Ko]
    node.attrs["ne16_weight_sum"] = gs.Constant(f"{name}_weight_sum", w_sum.astype(np.int32))

    # Pack weights into NE16 bitplane format — create NEW tensor to avoid breaking other nodes
    # Flatten to 2D [Ko, Ki] to stay compatible with GEMMLayer.computeShapes and tiling
    ne16_weights = _ne16_conv_1x1_weight_layout(w_int8)
    ne16_weights_2d = ne16_weights.reshape(Ko, -1)
    packedWeightTensor = gs.Constant(f"{name}_{weightTensor.name}", ne16_weights_2d)
    node.inputs[1] = packedWeightTensor

    # For RequantizedGemm: transform mul → ne16_scale, create scale_n, pre-multiply bias
    # All sign-independent — signed input compensation done at runtime in the template
    if node.op == 'RequantizedGemm' and len(node.inputs) >= 4:
        mulTensor = node.inputs[3]
        biasTensor = node.inputs[2]

        if isinstance(mulTensor, gs.Constant) and isinstance(biasTensor, gs.Constant):
            mul_values = mulTensor.values.flatten().astype(np.int32)
            log2D = int(np.log2(node.attrs['div'].values))

            # Broadcast scalar mul to per-channel if needed
            if len(mul_values) == 1:
                mul_values = np.full(Ko, mul_values[0], dtype=np.int32)

            ne16_scale, ne16_scale_n = _compute_ne16_scale_shift(mul_values, log2D)

            # Rescale bias from mul/log2D domain to scale/scale_n domain
            # bias_merged is already *= mul from PULPGEMMRequantMergePass
            # NE16 needs: bias_ne16 = bias_merged * 2^(scale_n - log2D)
            bias_values = biasTensor.values.flatten().astype(np.int64)
            ne16_bias = np.zeros(Ko, dtype=np.int64)
            for ko in range(Ko):
                shift_diff = int(ne16_scale_n[ko]) - log2D
                if shift_diff >= 0:
                    ne16_bias[ko] = bias_values[ko] << shift_diff
                else:
                    ne16_bias[ko] = bias_values[ko] >> (-shift_diff)

            # Signed input compensation: subtract 128 * w_sum * scale from bias
            input_signed = _is_input_signed(node, graph)
            if input_signed:
                for ko in range(Ko):
                    ne16_bias[ko] -= 128 * int(w_sum[ko]) * int(ne16_scale[ko])

            ne16_bias = ne16_bias.astype(np.int32)

            # Overwrite mul tensor with ne16_scale
            mulTensor.values = ne16_scale

            # Overwrite bias tensor
            biasTensor.values = ne16_bias

            # Append scale_n as new input[4]
            scale_n_tensor = gs.Constant(f"{name}_scale_n", ne16_scale_n)
            node.inputs.append(scale_n_tensor)

    elif node.op == 'Gemm' and len(node.inputs) >= 3:
        # Plain Gemm (int32 output): bias compensation for signed input (no scale)
        biasTensor = node.inputs[2]
        if isinstance(biasTensor, gs.Constant):
            input_signed = _is_input_signed(node, graph)
            if input_signed:
                bias_values = biasTensor.values.flatten().astype(np.int64)
                bias_values = bias_values - 128 * w_sum
                biasTensor.values = bias_values.astype(np.int32)

    return graph


@contextagnostic
class NE16AdjustGEMMWeightLayoutPass(ReplaceSequentialPatternPass):

    def __init__(self):
        graph = gs.Graph()
        _input = gs.Variable(name = 'input_1')
        output = graph.layer(inputs = [_input], outputs = ['out'], op = 'RequantizedGemm|Gemm', name = 'node')
        graph.outputs.append(output)
        graph.inputs.append(_input)

        super().__init__(graph, _ne16_adjust_gemm_weight_layout_fun,
                         "_NE16_ADJUST_GEMM_WEIGHT_LAYOUT_PASS",
                         NonBranchingMatcher(regex_op = True))
