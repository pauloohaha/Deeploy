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

      in_net_Buffer = ctxt.lookup(operatorRepresentation['data_in_net'])
      in_kk_Buffer  = ctxt.lookup(operatorRepresentation['data_in_kk'])
      out_net_Buffer = ctxt.lookup(operatorRepresentation['data_out'])
      out_net_Buffer._alias = in_net_Buffer.name

      return ctxt, operatorRepresentation, []
    
    
    def hoistTransientBuffers(self, ctxt: NetworkContext,
                              operatorRepresentation: OperatorRepresentation) -> Tuple[NetworkContext, Dict, List[str]]:
      
      MAX_EDGE_PER_PATCH = 5
      DIM = 384

      L1_edge_in_buffer_name  = operatorRepresentation['nodeName'] + "_L1_edge_in_buffer"
      L1_edge_out_buffer_name = operatorRepresentation['nodeName'] + "_L1_edge_out_buffer"
      L1_collected_edge_buffer_name = operatorRepresentation['nodeName'] + "_L1_collected_edge_buffer"

      L1_edge_buffer_dim = MAX_EDGE_PER_PATCH * DIM * 4 #float32
      L1_collected_edge_buffer_size = MAX_EDGE_PER_PATCH * 4 #int32

      ctxt.hoistTransientBuffer(L1_edge_in_buffer_name, L1_edge_buffer_dim)
      ctxt.hoistTransientBuffer(L1_edge_out_buffer_name, L1_edge_buffer_dim)
      ctxt.hoistTransientBuffer(L1_collected_edge_buffer_name, L1_collected_edge_buffer_size)
      
      ctxt.lookup(L1_edge_in_buffer_name)._type.referencedType = ctxt.lookup(
            operatorRepresentation['data_in_net'])._type.referencedType
      ctxt.lookup(L1_edge_out_buffer_name)._type.referencedType = ctxt.lookup(
            operatorRepresentation['data_out'])._type.referencedType
      ctxt.lookup(L1_collected_edge_buffer_name)._type.referencedType = ctxt.lookup(
            operatorRepresentation['data_in_kk'])._type.referencedType
      
      operatorRepresentation['edge_buff_l1_in'] = L1_edge_in_buffer_name
      operatorRepresentation['edge_buff_l1_out'] = L1_edge_out_buffer_name
      operatorRepresentation['collected_edge_id'] = L1_collected_edge_buffer_name

      return ctxt, operatorRepresentation, [L1_edge_in_buffer_name, L1_edge_out_buffer_name, L1_collected_edge_buffer_name]


referenceTemplate = CustomSoftmaxAgg("""
// Customized SoftMax agg for DPVO (Name: ${nodeName}, Op: ${nodeOp})

                                     
SoftMaxAgg_master_kernel( (float *)${data_in_net}, \
                          (int   *)${data_in_kk}, \
                          (float *)${data_out}, \
                          (float *)${edge_buff_l1_in}, \
                          (float *)${edge_buff_l1_out}, \
                          (int   *)${collected_edge_id});

    
                              
""")