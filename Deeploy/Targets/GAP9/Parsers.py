# SPDX-FileCopyrightText: 2023 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0

import math
from typing import Tuple
import numpy as np
import onnx_graphsurgeon as gs

from Deeploy.DeeployTypes import NetworkContext, NodeParser

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