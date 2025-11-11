# SPDX-FileCopyrightText: 2021 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0

from typing import Dict, List, Tuple

from Deeploy.DeeployTypes import NetworkContext, NodeTemplate, OperatorRepresentation, VariableBuffer


class CustomNop(NodeTemplate):
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

      inBuffer = ctxt.lookup(operatorRepresentation['data_in'])
      outBuffer = ctxt.lookup(operatorRepresentation['data_out'])
      outBuffer._alias = inBuffer.name

      return ctxt, operatorRepresentation, []
    

referenceTemplate = CustomNop("""
// Nop (Name: ${nodeName}, Op: ${nodeOp})
// Debug *** Piao
${data_out} = ${data_in};
                              
""")