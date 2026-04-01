# SPDX-FileCopyrightText: 2025 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0

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
        # transB=1: weight is [Ko, Ki] already
        Ko, Ki = values.shape
    else:
        # transB=0: weight is [Ki, Ko], need to transpose for NE16 packing
        Ki, Ko = values.shape
        values = values.T  # now [Ko, Ki]
        # Update transB since we transposed the weights
        node.attrs['transB'] = 1

    # Skip if Ki isn't NE16-compatible
    if Ki % 16 != 0:
        return graph

    # Compute per-channel weight sum BEFORE packing (needed for signed input bias compensation)
    w_int8 = values.astype(np.int8)
    w_sum = w_int8.astype(np.int64).sum(axis=1)  # [Ko]
    node.attrs["ne16_weight_sum"] = gs.Constant(f"{name}_weight_sum", w_sum.astype(np.int32))

    # Pack weights into NE16 bitplane format — create NEW tensor to avoid breaking other nodes
    # Flatten to 2D [Ko, Ki] to stay compatible with GEMMLayer.computeShapes and tiling
    # (Nb_KI * Qw * 2 == Ki when Ki % 16 == 0, so DMA offsets are correct)
    ne16_weights = _ne16_conv_1x1_weight_layout(w_int8)
    ne16_weights_2d = ne16_weights.reshape(Ko, -1)
    packedWeightTensor = gs.Constant(f"{name}_{weightTensor.name}", ne16_weights_2d)
    node.inputs[1] = packedWeightTensor

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
