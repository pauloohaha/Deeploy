# SPDX-FileCopyrightText: 2025 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0
"""Test vectors for networks generated with integer IO (GAP9 strip passes).

With DEEPLOY_STRIP_INPUT_QUANT / DEEPLOY_STRIP_OUTPUT_DEQUANT the network's graph
inputs/outputs become integer, while inputs.npz / outputs.npz hold the float values. This converts them the way
the producer / consumer outside the network does, so the input types are inferred as integers and the expected
outputs are integers:
  stripped input Quant:  q = trunc(clip(x * scale + zero_point + 0.5))   (as the SDK Quant kernel)
  stripped output:       q = clip(round(y / scale) + zero_point)           (the inverse of its Dequant)
Inputs that already hold integers are left unchanged."""

import os
from typing import List

import numpy as np
import onnx_graphsurgeon as gs


def _indices(var: str) -> List[int]:
    return [int(v) for v in os.environ.get(var, "").split(",") if v.strip() != ""]


def _attr(node: gs.Node, name: str, default = None):
    value = node.attrs.get(name, default)
    if isinstance(value, gs.Constant):
        value = value.values
    return np.array(value).item() if value is not None else None


def _is_integer(values: np.ndarray) -> bool:
    return bool(np.all(values == np.round(values)))


def _range(signed: bool, bit_width: int):
    return (-(2**(bit_width - 1)), 2**(bit_width - 1) - 1) if signed else (0, 2**bit_width - 1)


def convertIntegerIO(graph: gs.Graph, test_inputs: List[np.ndarray], test_outputs: List[np.ndarray] = None):
    strip_in = _indices("DEEPLOY_STRIP_INPUT_QUANT")
    strip_out = _indices("DEEPLOY_STRIP_OUTPUT_DEQUANT")
    if not (strip_in or strip_out):
        return test_inputs, test_outputs

    from Deeploy.Targets.Generic.TopologyOptimizationPasses.Passes import DequantPatternPass, QuantPatternPass
    g = graph.copy()
    g.fold_constants().cleanup()
    g = QuantPatternPass().apply(g)
    g = DequantPatternPass().apply(g)

    test_inputs = list(test_inputs)
    for idx in strip_in:
        x = test_inputs[idx]
        if _is_integer(x):
            continue
        quant = [n for n in g.inputs[idx].outputs if n.op == 'Quant']
        assert len(quant) == 1, f"input {idx} must feed one Quant"
        q = quant[0]
        scale, zp = np.float32(_attr(q, 'scale')), np.float32(_attr(q, 'zero_point', 0.0))
        lo, hi = _range(bool(_attr(q, 'signed', 1)), int(_attr(q, 'bit_width', 8)))
        v = (x.astype(np.float32).astype(np.float64) * np.float64(scale) + np.float64(zp + np.float32(0.5)))
        v = np.clip(v.astype(np.float32), lo, hi)
        test_inputs[idx] = np.trunc(v).astype(np.float64) + 0.0  # no negative zeros

    if test_outputs is not None:
        test_outputs = list(test_outputs)
        for idx in strip_out:
            y = test_outputs[idx]
            if _is_integer(y):
                continue
            dq = g.outputs[idx].inputs[0]
            assert dq.op == 'Dequant', f"output {idx} must be produced by a Dequant"
            scale, zp = float(_attr(dq, 'scale')), float(_attr(dq, 'zero_point', 0.0))
            lo, hi = _range(bool(_attr(dq, 'signed', 1)), int(_attr(dq, 'bit_width', 8)))
            test_outputs[idx] = np.clip(np.round(y / scale) + zp, lo, hi).astype(np.float64) + 0.0

    return test_inputs, test_outputs
