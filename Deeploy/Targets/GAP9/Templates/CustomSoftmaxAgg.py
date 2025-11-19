# SPDX-FileCopyrightText: 2025 ETH Zurich and University of Bologna
# Author: Pu Deng, ETH Zurich
# SPDX-License-Identifier: Apache-2.0

from typing import Dict, List, Tuple

from Deeploy.DeeployTypes import NetworkContext, NodeTemplate, OperatorRepresentation, VariableBuffer


class CustomSoftmaxAgg(NodeTemplate):
    def alignToContext(self, ctxt: NetworkContext,
                      operatorRepresentation: OperatorRepresentation) -> Tuple[NetworkContext, Dict, List[str]]:
      
      # SCHEREMO: Selectively mark 'indices' dead, since we don't need them
      if 'indices' in operatorRepresentation.keys():
          ctxt.globalObjects[operatorRepresentation['indices']]._deploy = False
          ctxt.globalObjects[operatorRepresentation['indices']]._live = False

      # Same for "shape"
      if "shape" in operatorRepresentation.keys():
          ctxt.globalObjects[operatorRepresentation["shape"]]._deploy = False
          ctxt.globalObjects[operatorRepresentation["shape"]]._live = False

      in_net_Buffer = ctxt.lookup(operatorRepresentation['data_in_1'])
      in_kk_Buffer  = ctxt.lookup(operatorRepresentation['data_in_2'])
      out_net_Buffer = ctxt.lookup(operatorRepresentation['data_out'])
      out_net_Buffer._alias = in_net_Buffer.name

      return ctxt, operatorRepresentation, []
    

referenceTemplate = CustomSoftmaxAgg("""
// Nop (Name: ${nodeName}, Op: ${nodeOp})
// Debug *** 
${data_out} = ${data_in_1};
                              
""")