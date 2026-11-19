# SPDX-FileCopyrightText: 2025 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0

import math
from functools import partial

import numpy as np
import onnx_graphsurgeon as gs

from Deeploy.CommonExtensions.OptimizationPasses.Matchers import Match, NonBranchingMatcher
from Deeploy.CommonExtensions.OptimizationPasses.PassClasses import Pass, ReplaceSequentialPatternPass, \
    contextagnostic


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


def _ne16_adjust_gemm_weight_layout_fun(graph: gs.Graph, match: Match, name: str):
    """Prepare GEMM node for NE16 execution.

    Handles transB normalization, scale/scale_n computation, and bias rescaling.
    Weight bitplane packing and signed bias compensation are deferred to alignToContext
    where input signedness is known from the type system.
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

    # Transpose weight to [Ko, Ki] if needed — keep as int8
    if not transB:
        transposed = values.T.astype(np.int8)
        newWeightTensor = gs.Constant(f"{name}_{weightTensor.name}", transposed)
        node.inputs[1] = newWeightTensor
        node.attrs['transB'] = 1

    # For RequantizedGemm: transform mul → ne16_scale, create scale_n, rescale bias
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

            ne16_bias = ne16_bias.astype(np.int32)

            # Overwrite mul tensor with ne16_scale
            mulTensor.values = ne16_scale

            # Overwrite bias tensor
            biasTensor.values = ne16_bias

            # Append scale_n as new input[4]
            scale_n_tensor = gs.Constant(f"{name}_scale_n", ne16_scale_n)
            node.inputs.append(scale_n_tensor)

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


# ---------------------------------------------------------------------------------------------------------------------
# Quantization boundary fusion (int8 TinyDEVO pipeline)
#
# The passes below remove Quant / Dequant nodes by moving them into the graph boundary or into a neighbouring kernel,
# without changing what is computed. They are opt-in through environment variables, read when the pass runs:
#   DEEPLOY_STRIP_INPUT_QUANT=<i>[,<j>...]      graph inputs whose only consumer is a Quant: the input becomes the
#                                              Quant's integer output (the caller quantizes)
#   DEEPLOY_STRIP_OUTPUT_DEQUANT=<i>[,<j>...]   graph outputs produced by a Dequant: the output becomes the Dequant's
#                                              integer input (the caller dequantizes)
#   DEEPLOY_FUSE_QUANT=1                       fuse Dequant -> LayerNormalization -> Quant into
#                                              RequantizedLayerNormalization
#   DEEPLOY_IO_SCALES_JSON=<path>              append the quantization of every stripped input/output to this file
# ---------------------------------------------------------------------------------------------------------------------

import json
import os


def _env_indices(var: str):
    value = os.environ.get(var, "").strip()
    return [int(v) for v in value.split(",") if v.strip() != ""]


def _attr(node: gs.Node, name: str, default = None):
    if name not in node.attrs:
        return default
    value = node.attrs[name]
    if isinstance(value, gs.Constant):
        value = value.values
    return np.array(value).item()


def _record_io_scale(entry: dict):
    path = os.environ.get("DEEPLOY_IO_SCALES_JSON", "")
    if path == "":
        return
    entries = []
    if os.path.exists(path):
        with open(path) as f:
            entries = json.load(f)
    entries = [e for e in entries if e["tensor"] != entry["tensor"]] + [entry]
    with open(path, "w") as f:
        json.dump(entries, f, indent = 2)


@contextagnostic
class GAP9StripGraphInputQuantPass(Pass):
    """graph input -> Quant  =>  integer graph input (the producer outside the network writes the quantized values)."""

    def __init__(self):
        super().__init__()

    def run_pass(self, graph: gs.Graph) -> gs.Graph:
        indices = _env_indices("DEEPLOY_STRIP_INPUT_QUANT")
        for idx in indices:
            graph_input = graph.inputs[idx]
            consumers = list(graph_input.outputs)
            assert len(consumers) == 1 and consumers[0].op == 'Quant', \
                f"input {idx} ({graph_input.name}) must feed exactly one Quant, found {[n.op for n in consumers]}"
            quant = consumers[0]
            quant_out = quant.outputs[0]
            for node in list(quant_out.outputs):
                node.inputs = [graph_input if t is quant_out else t for t in node.inputs]
            quant.inputs.clear()
            quant.outputs.clear()
            signed = bool(_attr(quant, 'signed', 1))
            bit_width = int(_attr(quant, 'bit_width', 8))
            graph_input.dtype = {(True, 8): np.int8, (False, 8): np.uint8, (True, 16): np.int16}[(signed, bit_width)]
            _record_io_scale({
                "tensor": graph_input.name,
                "kind": "input",
                "node": quant.name,
                # Quant: q = clip(x * scale + zero_point), scale is the multiplier (1 / step)
                "quant_scale": float(_attr(quant, 'scale')),
                "zero_point": float(_attr(quant, 'zero_point', 0.0)),
                "signed": signed,
                "bit_width": bit_width,
            })
        if indices:
            graph.cleanup().toposort()
        return graph


@contextagnostic
class GAP9StripGraphOutputDequantPass(Pass):
    """Dequant -> graph output  =>  integer graph output (the consumer outside the network dequantizes).

    A Quant consuming the dequantized tensor becomes a GAP9DequantQuant reading the integer output: it computes
    Dequant -> Quant in fp32 exactly as today (an integer RequantShift from DequantQuantMergePass can differ by
    one step). Other consumers keep the Dequant."""

    def __init__(self):
        super().__init__()

    def run_pass(self, graph: gs.Graph) -> gs.Graph:
        indices = _env_indices("DEEPLOY_STRIP_OUTPUT_DEQUANT")
        for idx in indices:
            float_out = graph.outputs[idx]
            dequant = float_out.inputs[0] if len(float_out.inputs) == 1 else None
            assert dequant is not None and dequant.op == 'Dequant', \
                f"output {idx} ({float_out.name}) must be produced by a Dequant"
            int_out = dequant.inputs[0]
            out_name = float_out.name
            float_out.name = f"{out_name}_dequantized"
            int_out.name = out_name
            graph.outputs[idx] = int_out
            for quant in [n for n in list(float_out.outputs) if n.op == 'Quant']:
                fused = gs.Node(op = 'GAP9DequantQuant',
                                name = f"{quant.name}_from_{out_name}",
                                attrs = {
                                    'in_scale': float(_attr(dequant, 'scale')),
                                    'in_zero_point': float(_attr(dequant, 'zero_point', 0.0)),
                                    'out_scale': float(_attr(quant, 'scale')),
                                    'out_zero_point': float(_attr(quant, 'zero_point', 0.0)),
                                    'out_signed': int(bool(_attr(quant, 'signed', 1))),
                                    'out_bit_width': int(_attr(quant, 'bit_width', 8)),
                                },
                                inputs = [int_out],
                                outputs = list(quant.outputs))
                quant.inputs.clear()
                quant.outputs.clear()
                graph.nodes.append(fused)
            if len(float_out.outputs) == 0:
                dequant.inputs.clear()
                dequant.outputs.clear()
            _record_io_scale({
                "tensor": out_name,
                "kind": "output",
                "node": dequant.name,
                # Dequant: x = (q - zero_point) * scale, scale is the step
                "dequant_scale": float(_attr(dequant, 'scale')),
                "zero_point": float(_attr(dequant, 'zero_point', 0.0)),
            })
        if indices:
            graph.cleanup().toposort()
        return graph


def _rq_layernorm_fun(graph: gs.Graph, match: Match, name: str):
    dequant, layernorm, quant = [m for k, m in match.nodes_map.items()]
    if os.environ.get("DEEPLOY_FUSE_QUANT", "0") != "1":
        return graph
    assert layernorm.inputs[0] is dequant.outputs[0]
    attrs = {
        'epsilon': float(_attr(layernorm, 'epsilon')),
        'in_scale': float(_attr(dequant, 'scale')),
        'in_zero_point': float(_attr(dequant, 'zero_point', 0.0)),
        'out_scale': float(_attr(quant, 'scale')),
        'out_zero_point': float(_attr(quant, 'zero_point', 0.0)),
        'out_signed': int(bool(_attr(quant, 'signed', 1))),
        'out_bit_width': int(_attr(quant, 'bit_width', 8)),
    }
    _inputs = [dequant.inputs[0], layernorm.inputs[1], layernorm.inputs[2]]
    _outputs = quant.outputs
    rq_layernorm = gs.Node(op = 'RequantizedLayerNormalization', name = layernorm.name, attrs = attrs)
    graph.replaceInsertNode(_inputs, _outputs, rq_layernorm)
    return graph


@contextagnostic
class GAP9RequantizedLayerNormPass(ReplaceSequentialPatternPass):
    """Dequant -> LayerNormalization -> Quant  =>  RequantizedLayerNormalization (bit-identical fused kernel)."""

    def __init__(self):
        graph = gs.Graph()
        _input = gs.Variable(name = 'input_1')
        output = graph.layer(inputs = [_input], outputs = ['dequant_out'], op = 'Dequant', name = 'dequant')
        output = graph.layer(inputs = output, outputs = ['layernorm_out'], op = 'LayerNormalization', name = 'layernorm')
        output = graph.layer(inputs = output, outputs = ['quant_out'], op = 'Quant', name = 'quant')
        graph.outputs.append(output)
        graph.inputs.append(_input)
        super().__init__(graph, _rq_layernorm_fun, "_GAP9_RQ_LAYERNORM_PASS")


def _uniform_rqs_attrs(rqs: gs.Node):
    """Scalar mul/add and attributes of a uniform RequantShift node, or None if it is per-channel."""
    mul, add = rqs.inputs[1], rqs.inputs[2]
    if not (isinstance(mul, gs.Constant) and isinstance(add, gs.Constant)):
        return None
    if mul.values.size != 1 or add.values.size != 1:
        return None
    div = int(_attr(rqs, 'div'))
    log2D = int(np.log2(div))
    assert 2**log2D == div, f"{rqs.name}: div {div} is not a power of two"
    return {
        'rqs_mul': int(mul.values.flatten()[0]),
        'rqs_add': int(add.values.flatten()[0]),
        'rqs_log2D': log2D,
        'n_levels': int(_attr(rqs, 'n_levels_out', _attr(rqs, 'n_levels', 256))),
        'signed': int(bool(_attr(rqs, 'signed', 1))),
    }


def _rqs_gather_fun(graph: gs.Graph, match: Match, name: str):
    rqs, gather = [m for k, m in match.nodes_map.items()]
    if os.environ.get("DEEPLOY_FUSE_QUANT", "0") != "1":
        return graph
    attrs = _uniform_rqs_attrs(rqs)
    if attrs is None or gather.inputs[0] is not rqs.outputs[0]:
        return graph
    attrs['dir'] = gather.attrs['dir']
    fused = gs.Node(op = 'RequantizedTCneighborGather', name = gather.name, attrs = attrs)
    graph.replaceInsertNode([rqs.inputs[0], gather.inputs[1]], gather.outputs, fused)
    return graph


@contextagnostic
class GAP9RequantizedGatherPass(ReplaceSequentialPatternPass):
    """RequantShift -> TCneighborGather  =>  RequantizedTCneighborGather (requantize while gathering)."""

    def __init__(self):
        graph = gs.Graph()
        _input = gs.Variable(name = 'input_1')
        output = graph.layer(inputs = [_input], outputs = ['rqs_out'], op = 'RequantShift', name = 'rqs')
        output = graph.layer(inputs = output, outputs = ['gather_out'], op = 'TCneighborGather', name = 'gather')
        graph.outputs.append(output)
        graph.inputs.append(_input)
        super().__init__(graph, _rqs_gather_fun, "_GAP9_RQS_GATHER_PASS")


def _add_rqs_fun(graph: gs.Graph, match: Match, name: str):
    add, rqs = [m for k, m in match.nodes_map.items()]
    if os.environ.get("DEEPLOY_FUSE_QUANT", "0") != "1":
        return graph
    attrs = _uniform_rqs_attrs(rqs)
    if attrs is None or rqs.inputs[0] is not add.outputs[0] or len(add.inputs) != 2:
        return graph
    if any(isinstance(t, gs.Constant) for t in add.inputs):
        return graph
    fused = gs.Node(op = 'GAP9AddRequantShift', name = add.name, attrs = attrs)
    graph.replaceInsertNode(list(add.inputs), rqs.outputs, fused)
    return graph


@contextagnostic
class GAP9AddRequantShiftPass(ReplaceSequentialPatternPass):
    """Add -> RequantShift  =>  GAP9AddRequantShift (multi-core, no int16 intermediate)."""

    def __init__(self):
        graph = gs.Graph()
        _input = gs.Variable(name = 'input_1')
        output = graph.layer(inputs = [_input], outputs = ['add_out'], op = 'Add', name = 'add')
        output = graph.layer(inputs = output, outputs = ['rqs_out'], op = 'RequantShift', name = 'rqs')
        graph.outputs.append(output)
        graph.inputs.append(_input)
        super().__init__(graph, _add_rqs_fun, "_GAP9_ADD_RQS_PASS")


def _single_consumer(tensor: gs.Variable, op: str):
    consumers = list(tensor.outputs)
    if len(consumers) == 1 and consumers[0].op == op:
        return consumers[0]
    return None


def _producer(tensor: gs.Tensor, op: str):
    if isinstance(tensor, gs.Variable) and len(tensor.inputs) == 1 and tensor.inputs[0].op == op:
        return tensor.inputs[0]
    return None


def _quant_out_attrs(quant: gs.Node) -> dict:
    return {
        'out_scale': float(_attr(quant, 'scale')),
        'out_zero_point': float(_attr(quant, 'zero_point', 0.0)),
        'out_signed': int(bool(_attr(quant, 'signed', 1))),
        'out_bit_width': int(_attr(quant, 'bit_width', 8)),
    }


def _dequant_attrs(dequant: gs.Node, prefix: str) -> dict:
    return {
        f'{prefix}_scale': float(_attr(dequant, 'scale')),
        f'{prefix}_zero_point': float(_attr(dequant, 'zero_point', 0.0)),
    }


def _disconnect(*nodes: gs.Node):
    for node in nodes:
        node.inputs.clear()
        node.outputs.clear()


def _dequant_quant_node(dequant: gs.Node, quant: gs.Node) -> gs.Node:
    """Dequant -> Quant as GAP9DequantQuant (fp32, bit-identical to the SDK pair) reading the integer tensor."""
    return gs.Node(op = 'GAP9DequantQuant',
                   name = f"{quant.name}_from_int",
                   attrs = {
                       'in_scale': float(_attr(dequant, 'scale')),
                       'in_zero_point': float(_attr(dequant, 'zero_point', 0.0)),
                       'out_scale': float(_attr(quant, 'scale')),
                       'out_zero_point': float(_attr(quant, 'zero_point', 0.0)),
                       'out_signed': int(bool(_attr(quant, 'signed', 1))),
                       'out_bit_width': int(_attr(quant, 'bit_width', 8)),
                   },
                   inputs = [dequant.inputs[0]],
                   outputs = list(quant.outputs))


@contextagnostic
class GAP9AttentionInt8FusionPass(Pass):
    """TinyDEVO update block: fuse the float softmax aggregations and scatter adds into integer-IO kernels.

    1. Dequant(g), Dequant(f) -> CustomColSoftmax -> CustomElementMul -> CustomColSum -> Quant
         =>  GAP9ColAttnPool(g, f, kk)
    2. net = Dequant(n16); y = CustomColScatter(Dequant(h1), net); Quant(y) (one or more identical);
       z = CustomColScatter(Dequant(h2), y); Quant(LayerNorm(z))
         =>  GAP9ScatterQuant(n16, h1, kk), GAP9ScatterScatterRQLN(n16, h1, h2, kk, w, b);
       the other Quant consumers of net become GAP9DequantQuant(n16).
    All kernels recompute the float values from the integer tensors with the original operations, so the
    results are bit-identical. Active only with DEEPLOY_FUSE_QUANT=1."""

    def __init__(self):
        super().__init__()

    def run_pass(self, graph: gs.Graph) -> gs.Graph:
        if os.environ.get("DEEPLOY_FUSE_QUANT", "0") != "1":
            return graph
        changed = self._fuse_pools(graph)
        changed |= self._fuse_scatters(graph)
        if changed:
            graph.cleanup().toposort()
        return graph

    def _fuse_pools(self, graph: gs.Graph) -> bool:
        changed = False
        for softmax in [n for n in graph.nodes if n.op == 'CustomColSoftmax']:
            if len(softmax.inputs) != 2 or len(softmax.outputs) != 1:
                continue
            dq_g = _producer(softmax.inputs[0], 'Dequant')
            mul = _single_consumer(softmax.outputs[0], 'CustomElementMul')
            if dq_g is None or mul is None or len(dq_g.outputs[0].outputs) != 1:
                continue
            others = [t for t in mul.inputs if t is not softmax.outputs[0]]
            dq_f = _producer(others[0], 'Dequant') if len(others) == 1 else None
            colsum = _single_consumer(mul.outputs[0], 'CustomColSum')
            if dq_f is None or colsum is None or len(dq_f.outputs[0].outputs) != 1:
                continue
            quant = _single_consumer(colsum.outputs[0], 'Quant')
            if quant is None or int(_attr(colsum, 'dir')) != int(_attr(softmax, 'dir')):
                continue
            attrs = {'dir': int(_attr(softmax, 'dir'))}
            attrs.update(_dequant_attrs(dq_g, 'g'))
            attrs.update(_dequant_attrs(dq_f, 'f'))
            attrs.update(_quant_out_attrs(quant))
            fused = gs.Node(op = 'GAP9ColAttnPool',
                            name = f"{softmax.name}_attnpool",
                            attrs = attrs,
                            inputs = [dq_g.inputs[0], dq_f.inputs[0], softmax.inputs[1]],
                            outputs = list(quant.outputs))
            _disconnect(dq_g, dq_f, softmax, mul, colsum, quant)
            graph.nodes.append(fused)
            changed = True
        return changed

    def _fuse_scatters(self, graph: gs.Graph) -> bool:
        changed = False
        for scatter1 in [n for n in graph.nodes if n.op == 'CustomColScatter']:
            if len(scatter1.inputs) != 3 or len(scatter1.outputs) != 1:
                continue
            dq_h1 = _producer(scatter1.inputs[0], 'Dequant')
            dq_net = _producer(scatter1.inputs[1], 'Dequant')
            if dq_h1 is None or dq_net is None or len(dq_h1.outputs[0].outputs) != 1:
                continue
            y = scatter1.outputs[0]
            quants = [n for n in y.outputs if n.op == 'Quant']
            scatter2s = [n for n in y.outputs if n.op == 'CustomColScatter' and n.inputs[1] is y]
            if len(quants) == 0 or len(scatter2s) != 1 or len(quants) + 1 != len(y.outputs):
                continue
            q_attrs = [_quant_out_attrs(q) for q in quants]
            if any(a != q_attrs[0] for a in q_attrs):
                continue
            scatter2 = scatter2s[0]
            dq_h2 = _producer(scatter2.inputs[0], 'Dequant')
            layernorm = _single_consumer(scatter2.outputs[0], 'LayerNormalization')
            if dq_h2 is None or layernorm is None or len(dq_h2.outputs[0].outputs) != 1:
                continue
            q_ln = _single_consumer(layernorm.outputs[0], 'Quant')
            if q_ln is None or scatter2.inputs[2] is not scatter1.inputs[2]:
                continue
            net16, h1, h2, kk = dq_net.inputs[0], dq_h1.inputs[0], dq_h2.inputs[0], scatter1.inputs[2]

            # y -> Quant (duplicates share one output)
            attrs = {'dir': int(_attr(scatter1, 'dir'))}
            attrs.update(_dequant_attrs(dq_net, 'net'))
            attrs.update(_dequant_attrs(dq_h1, 'agg'))
            attrs.update(q_attrs[0])
            q_out = quants[0].outputs[0]
            for q in quants[1:]:
                for consumer in list(q.outputs[0].outputs):
                    consumer.inputs = [q_out if t is q.outputs[0] else t for t in consumer.inputs]
            fused_q = gs.Node(op = 'GAP9ScatterQuant',
                              name = f"{scatter1.name}_quant",
                              attrs = attrs,
                              inputs = [net16, h1, kk],
                              outputs = [q_out])

            # z -> LayerNorm -> Quant
            attrs = {'dir1': int(_attr(scatter1, 'dir')), 'dir2': int(_attr(scatter2, 'dir'))}
            attrs.update(_dequant_attrs(dq_net, 'net'))
            attrs.update(_dequant_attrs(dq_h1, 'agg1'))
            attrs.update(_dequant_attrs(dq_h2, 'agg2'))
            attrs['epsilon'] = float(_attr(layernorm, 'epsilon'))
            attrs.update(_quant_out_attrs(q_ln))
            fused_ln = gs.Node(op = 'GAP9ScatterScatterRQLN',
                               name = f"{layernorm.name}_scatter_rqln",
                               attrs = attrs,
                               inputs = [net16, h1, h2, kk, layernorm.inputs[1], layernorm.inputs[2]],
                               outputs = list(q_ln.outputs))

            _disconnect(dq_h1, dq_h2, scatter1, scatter2, layernorm, q_ln, *quants)
            graph.nodes.extend([fused_q, fused_ln])

            # the remaining consumers of the dequantized net: Quant -> fp32 DequantQuant from the integer tensor
            net_float = dq_net.outputs[0]
            for q in [n for n in list(net_float.outputs) if n.op == 'Quant']:
                graph.nodes.append(_dequant_quant_node(dq_net, q))
                _disconnect(q)
            if len(net_float.outputs) == 0:
                _disconnect(dq_net)
            changed = True
        return changed
