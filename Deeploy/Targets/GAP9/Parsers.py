# SPDX-FileCopyrightText: 2023 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0

import math
from typing import List, Tuple
import numpy as np
import onnx_graphsurgeon as gs
from Deeploy.Targets.GAP9.Templates.DPVO_defines import MAX_PATCH_PER_FRAME, MAX_EDGE_PER_PATCH, DIM
from Deeploy.DeeployTypes import NetworkContext, NodeParser
from Deeploy.Targets.Generic.Parsers import GEMMParser, RQSParserInterface

class CustomColSoftmaxParser(NodeParser):

    def __init__(self):
        super().__init__()

    def parseNode(self, node: gs.Node) -> (bool):
        self.operatorRepresentation['dir'] = node.attrs['dir']
        return True

    def parseNodeCtxt(self,
                      ctxt: NetworkContext,
                      node: gs.Node,
                      channels_first: bool = True) -> Tuple[NetworkContext, bool]:
        
        for tensor, symName in zip(node.inputs, ['data_in_net', 'data_in_kk']):
            self.operatorRepresentation[symName] = ctxt.lookup(tensor.name).name
        for tensor, symName in zip(node.outputs, ['data_out']):
            self.operatorRepresentation[symName] = ctxt.lookup(tensor.name).name
        

        return ctxt, True
    
class CustomColSumParser(NodeParser):

    def __init__(self):
        super().__init__()

    def parseNode(self, node: gs.Node) -> (bool):
        self.operatorRepresentation['dir'] = node.attrs['dir']
        return True

    def parseNodeCtxt(self,
                      ctxt: NetworkContext,
                      node: gs.Node,
                      channels_first: bool = True) -> Tuple[NetworkContext, bool]:
        
        for tensor, symName in zip(node.inputs, ['data_in_net', 'data_in_kk']):
            self.operatorRepresentation[symName] = ctxt.lookup(tensor.name).name
        for tensor, symName in zip(node.outputs, ['data_out']):
            self.operatorRepresentation[symName] = ctxt.lookup(tensor.name).name
        

        return ctxt, True


class CustomColScatterParser(NodeParser):

    def __init__(self):
        super().__init__()

    def parseNode(self, node: gs.Node) -> (bool):
        self.operatorRepresentation['dir'] = node.attrs['dir']
        return True

    def parseNodeCtxt(self,
                      ctxt: NetworkContext,
                      node: gs.Node,
                      channels_first: bool = True) -> Tuple[NetworkContext, bool]:
        
        for tensor, symName in zip(node.inputs, ['data_in_agg', 'data_in_net', 'data_in_kk']):
            self.operatorRepresentation[symName] = ctxt.lookup(tensor.name).name
        for tensor, symName in zip(node.outputs, ['data_out']):
            self.operatorRepresentation[symName] = ctxt.lookup(tensor.name).name
        

        return ctxt, True


class FloatSigmoidParser(NodeParser):

    def __init__(self):
        super().__init__()

    def parseNode(self, node: gs.Node) -> (bool):

        ret = all([len(node.inputs) == 1, len(node.outputs) == 1])
        
        return ret

    def parseNodeCtxt(self,
                      ctxt: NetworkContext,
                      node: gs.Node,
                      channels_first: bool = True) -> Tuple[NetworkContext, bool]:

        data_in = ctxt.lookup(node.inputs[0].name)
        data_out = ctxt.lookup(node.outputs[0].name)
        self.operatorRepresentation['data_in'] = data_in.name
        self.operatorRepresentation['data_out'] = data_out.name
        self.operatorRepresentation['size'] = np.prod(data_in.shape)

        return ctxt, True
    

class TCneighborGatherParser(NodeParser):

    def __init__(self):
        super().__init__()

    def parseNode(self, node: gs.Node) -> (bool):

        ret = all([len(node.inputs) == 2, len(node.outputs) == 1])
        self.operatorRepresentation['dir'] = node.attrs['dir']
        self.operatorRepresentation['dim'] = DIM
        return ret

    def parseNodeCtxt(self,
                      ctxt: NetworkContext,
                      node: gs.Node,
                      channels_first: bool = True) -> Tuple[NetworkContext, bool]:


        for tensor, symName in zip(node.inputs, ['data_in_net', 'data_in_kk']):
            self.operatorRepresentation[symName] = ctxt.lookup(tensor.name).name
        for tensor, symName in zip(node.outputs, ['data_out']):
            self.operatorRepresentation[symName] = ctxt.lookup(tensor.name).name
        
        return ctxt, True
    



class InstanceNorm2DParser(NodeParser):

    def __init__(self):
        super().__init__()

    def parseNode(self, node: gs.Node) -> (bool):

        ret = all(['epsilon' in node.attrs, len(node.inputs) == 3, len(node.outputs) >= 1])

        if ret:
            self.operatorRepresentation['epsilon'] = node.attrs['epsilon']

        return ret

    def parseNodeCtxt(self,
                      ctxt: NetworkContext,
                      node: gs.Node,
                      channels_first: bool = True) -> Tuple[NetworkContext, bool]:

        inputs = ['data_in', 'weight', 'bias']
        outputs = ['data_out']

        for idx, inputNode in enumerate(node.inputs):
            self.operatorRepresentation[inputs[idx]] = ctxt.lookup(inputNode.name).name
        for idx, outputNode in enumerate(node.outputs):
            self.operatorRepresentation[outputs[idx]] = ctxt.lookup(outputNode.name).name

        self.operatorRepresentation['size'] = np.prod(ctxt.lookup(node.inputs[0].name).shape)
        #instancenorm2d normalize across HxW, equivalent to instance norm but last dimlength is hxw
        self.operatorRepresentation['last2DimLength'] = ctxt.lookup(node.inputs[0].name).shape[-1] * ctxt.lookup(node.inputs[0].name).shape[-2]

        return ctxt, True


class NE16GEMMParser(GEMMParser, RQSParserInterface):
    """Parser for NE16 RequantizedGemm nodes with 5 inputs [A, B, C, mul, scale_n]."""

    def __init__(self):
        super().__init__(noBiasHoisting = True)

    def parseNode(self, node: gs.Node) -> bool:
        ret_rqs = RQSParserInterface.parseNode(self, node)
        ret_matmul = GEMMParser.parseNode(self, node)
        ret = all([ret_rqs, ret_matmul, 'shift' in node.attrs, len(node.inputs) == 5])
        if ret:
            self.operatorRepresentation['shift'] = int(node.attrs['shift'].values)
        return ret

    def parseNodeCtxt(self, ctxt: NetworkContext, node: gs.Node,
                      channels_first: bool = True) -> Tuple[NetworkContext, bool]:
        newCtxt, ret = GEMMParser.parseNodeCtxt(self, ctxt, node, channels_first)
        if ret:
            inputs = ['A', 'B', 'C', 'mul', 'scale_n']
            for idx, inputNode in enumerate(node.inputs):
                self.operatorRepresentation[inputs[idx]] = newCtxt.lookup(inputNode.name).name
            return newCtxt, True
        return ctxt, False


class RequantizedLayerNormParser(NodeParser):
    """Dequant -> LayerNormalization -> Quant fused by GAP9RequantizedLayerNormPass.

    Attributes (all from the original nodes): epsilon (LayerNorm), in_scale / in_zero_point (Dequant: x = (q - zp) * s),
    out_scale / out_zero_point / out_signed / out_bit_width (Quant: q = clip(x * scale + zp), scale = 1 / step).
    """

    _ATTRS = ['epsilon', 'in_scale', 'in_zero_point', 'out_scale', 'out_zero_point', 'out_signed', 'out_bit_width']

    def __init__(self):
        super().__init__()

    def parseNode(self, node: gs.Node) -> bool:
        ret = all([attr in node.attrs for attr in self._ATTRS]) and len(node.inputs) == 3 and len(node.outputs) == 1
        if ret:
            for attr in ['epsilon', 'in_scale', 'in_zero_point', 'out_scale', 'out_zero_point']:
                self.operatorRepresentation[attr] = float(np.array(node.attrs[attr]).item())
            self.operatorRepresentation['out_signed'] = bool(np.array(node.attrs['out_signed']).item())
            self.operatorRepresentation['out_bit_width'] = int(np.array(node.attrs['out_bit_width']).item())
        return ret

    def parseNodeCtxt(self,
                      ctxt: NetworkContext,
                      node: gs.Node,
                      channels_first: bool = True) -> Tuple[NetworkContext, bool]:
        for idx, name in enumerate(['data_in', 'weight', 'bias']):
            self.operatorRepresentation[name] = ctxt.lookup(node.inputs[idx].name).name
        self.operatorRepresentation['data_out'] = ctxt.lookup(node.outputs[0].name).name
        self.operatorRepresentation['size'] = int(np.prod(ctxt.lookup(node.inputs[0].name).shape))
        self.operatorRepresentation['lastDimLength'] = ctxt.lookup(node.inputs[0].name).shape[-1]
        return ctxt, True


def _parse_requant_attrs(node: gs.Node, operatorRepresentation: dict) -> bool:
    """Attributes of a fused uniform RequantShift (from GAP9 fusion passes)."""
    keys = ['rqs_mul', 'rqs_add', 'rqs_log2D', 'n_levels', 'signed']
    if not all(k in node.attrs for k in keys):
        return False
    operatorRepresentation['rqs_mul'] = int(np.array(node.attrs['rqs_mul']).item())
    operatorRepresentation['rqs_add'] = int(np.array(node.attrs['rqs_add']).item())
    operatorRepresentation['log2D'] = int(np.array(node.attrs['rqs_log2D']).item())
    operatorRepresentation['n_levels'] = int(np.array(node.attrs['n_levels']).item())
    operatorRepresentation['signed'] = bool(np.array(node.attrs['signed']).item())
    if operatorRepresentation['signed']:
        operatorRepresentation['output_min'] = -(operatorRepresentation['n_levels'] // 2)
        operatorRepresentation['output_max'] = (operatorRepresentation['n_levels'] // 2) - 1
    else:
        operatorRepresentation['output_min'] = 0
        operatorRepresentation['output_max'] = operatorRepresentation['n_levels'] - 1
    return True


class RequantizedTCneighborGatherParser(TCneighborGatherParser):
    """RequantShift -> TCneighborGather fused by GAP9RequantizedGatherPass."""

    def parseNode(self, node: gs.Node) -> bool:
        ret = super().parseNode(node)
        return ret and _parse_requant_attrs(node, self.operatorRepresentation)


class AddRequantShiftParser(NodeParser):
    """Add -> RequantShift fused by GAP9AddRequantShiftPass."""

    def __init__(self):
        super().__init__()

    def parseNode(self, node: gs.Node) -> bool:
        ret = len(node.inputs) == 2 and len(node.outputs) == 1
        return ret and _parse_requant_attrs(node, self.operatorRepresentation)

    def parseNodeCtxt(self,
                      ctxt: NetworkContext,
                      node: gs.Node,
                      channels_first: bool = True) -> Tuple[NetworkContext, bool]:
        data_in_1 = ctxt.lookup(node.inputs[0].name)
        data_in_2 = ctxt.lookup(node.inputs[1].name)
        data_out = ctxt.lookup(node.outputs[0].name)
        if tuple(data_in_1.shape) != tuple(data_in_2.shape):
            return ctxt, False
        self.operatorRepresentation['data_in_1'] = data_in_1.name
        self.operatorRepresentation['data_in_2'] = data_in_2.name
        self.operatorRepresentation['data_out'] = data_out.name
        self.operatorRepresentation['size'] = int(np.prod(data_in_1.shape))
        return ctxt, True


class DequantQuantParser(NodeParser):
    """Dequant -> Quant kept in fp32 (GAP9StripGraphOutputDequantPass), bit-identical to the SDK pair."""

    def __init__(self):
        super().__init__()

    def parseNode(self, node: gs.Node) -> bool:
        keys = ['in_scale', 'in_zero_point', 'out_scale', 'out_zero_point', 'out_signed', 'out_bit_width']
        if not (len(node.inputs) == 1 and len(node.outputs) == 1 and all(k in node.attrs for k in keys)):
            return False
        for k in keys[:4]:
            self.operatorRepresentation[k] = float(np.array(node.attrs[k]).item())
        self.operatorRepresentation['out_signed'] = int(np.array(node.attrs['out_signed']).item())
        self.operatorRepresentation['out_bit_width'] = int(np.array(node.attrs['out_bit_width']).item())
        return True

    def parseNodeCtxt(self,
                      ctxt: NetworkContext,
                      node: gs.Node,
                      channels_first: bool = True) -> Tuple[NetworkContext, bool]:
        data_in = ctxt.lookup(node.inputs[0].name)
        data_out = ctxt.lookup(node.outputs[0].name)
        self.operatorRepresentation['data_in'] = data_in.name
        self.operatorRepresentation['data_out'] = data_out.name
        self.operatorRepresentation['size'] = int(np.prod(data_in.shape))
        return ctxt, True


class _FusedIntParser(NodeParser):
    """Parser of the fused integer-IO nodes from GAP9AttentionInt8FusionPass: inputs in order, scalar attributes."""

    inputKeys: List[str] = []
    floatAttrs: List[str] = []
    intAttrs: List[str] = []

    def __init__(self):
        super().__init__()

    def parseNode(self, node: gs.Node) -> bool:
        if len(node.inputs) != len(self.inputKeys) or len(node.outputs) != 1:
            return False
        if not all(k in node.attrs for k in self.floatAttrs + self.intAttrs):
            return False
        for k in self.floatAttrs:
            self.operatorRepresentation[k] = float(np.array(node.attrs[k]).item())
        for k in self.intAttrs:
            self.operatorRepresentation[k] = int(np.array(node.attrs[k]).item())
        return True

    def parseNodeCtxt(self,
                      ctxt: NetworkContext,
                      node: gs.Node,
                      channels_first: bool = True) -> Tuple[NetworkContext, bool]:
        for key, tensor in zip(self.inputKeys, node.inputs):
            self.operatorRepresentation[key] = ctxt.lookup(tensor.name).name
        self.operatorRepresentation['data_out'] = ctxt.lookup(node.outputs[0].name).name
        return ctxt, True


_OUT_ATTRS = ['out_scale', 'out_zero_point']
_OUT_INT_ATTRS = ['out_signed', 'out_bit_width']


class ColAttnPoolParser(_FusedIntParser):
    inputKeys = ['data_in_g', 'data_in_f', 'data_in_kk']
    floatAttrs = ['g_scale', 'g_zero_point', 'f_scale', 'f_zero_point'] + _OUT_ATTRS
    intAttrs = ['dir'] + _OUT_INT_ATTRS


class ScatterQuantParser(_FusedIntParser):
    inputKeys = ['data_in_net', 'data_in_agg', 'data_in_kk']
    floatAttrs = ['net_scale', 'net_zero_point', 'agg_scale', 'agg_zero_point'] + _OUT_ATTRS
    intAttrs = ['dir'] + _OUT_INT_ATTRS


class ScatterScatterRQLNParser(_FusedIntParser):
    inputKeys = ['data_in_net', 'data_in_agg1', 'data_in_agg2', 'data_in_kk', 'weight', 'bias']
    floatAttrs = ['net_scale', 'net_zero_point', 'agg1_scale', 'agg1_zero_point', 'agg2_scale', 'agg2_zero_point',
                  'epsilon'] + _OUT_ATTRS
    intAttrs = ['dir1', 'dir2'] + _OUT_INT_ATTRS
