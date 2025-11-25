# SPDX-FileCopyrightText: 2023 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0

import math
from typing import Tuple
import numpy as np
import onnx_graphsurgeon as gs

from Deeploy.DeeployTypes import NetworkContext, NodeParser

class CustomSoftmaxAggParser(NodeParser):

    def __init__(self):
        super().__init__()

    def parseNode(self, node: gs.Node) -> (bool):
        pass
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
