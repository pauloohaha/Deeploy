# SPDX-FileCopyrightText: 2025 ETH Zurich and University of Bologna
# Author: Pu Deng, ETH Zurich
# SPDX-License-Identifier: Apache-2.0

from typing import Dict, List, Tuple

from Deeploy.DeeployTypes import NetworkContext, NodeTemplate, OperatorRepresentation, VariableBuffer
from Deeploy.Targets.GAP9.Templates.DPVO_defines import MAX_PATCH_PER_FRAME, MAX_EDGE_PER_PATCH, DIM

class CustomnColSum(NodeTemplate):
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

      #expect the input and output buffers in L2
      in_net_Buffer._targetmemorylevel['CustomnColSum'] = 'L2'
      in_kk_Buffer._targetmemorylevel['CustomnColSum'] = 'L2'
      out_net_Buffer._targetmemorylevel['CustomnColSum'] = 'L2'
      out_net_Buffer._alias = in_net_Buffer.name

      return ctxt, operatorRepresentation, []
    
    
    def hoistTransientBuffers(self, ctxt: NetworkContext,
                              operatorRepresentation: OperatorRepresentation) -> Tuple[NetworkContext, Dict, List[str]]:
      


      L1_edge_in_ping_buffer_name         = operatorRepresentation['nodeName'] + "_L1_edge_in_ping_buffer"
      L1_edge_in_pong_buffer_name         = operatorRepresentation['nodeName'] + "_L1_edge_in_pong_buffer"
      L1_edge_out_ping_buffer_name        = operatorRepresentation['nodeName'] + "_L1_edge_out_ping_buffer"
      L1_edge_out_pong_buffer_name        = operatorRepresentation['nodeName'] + "_L1_edge_out_pong_buffer"

      if operatorRepresentation['dir'] == 0:
          # patch aggregation, process MAX_EDGE_PER_PATCH from one patch each time
          L1_edge_buffer_dim = MAX_EDGE_PER_PATCH * DIM * 4 #float32
      else:
          # frame aggregation, process MAX_PATCH_PER_FRAME from one frame each time
          L1_edge_buffer_dim = MAX_PATCH_PER_FRAME * DIM * 4 #float32

      ctxt.hoistTransientBuffer(L1_edge_in_ping_buffer_name, L1_edge_buffer_dim)
      ctxt.hoistTransientBuffer(L1_edge_out_ping_buffer_name, L1_edge_buffer_dim)
      ctxt.hoistTransientBuffer(L1_edge_in_pong_buffer_name, L1_edge_buffer_dim)
      ctxt.hoistTransientBuffer(L1_edge_out_pong_buffer_name, L1_edge_buffer_dim)
      
      ctxt.lookup(L1_edge_in_ping_buffer_name)._type.referencedType = ctxt.lookup(
            operatorRepresentation['data_in_net'])._type.referencedType
      ctxt.lookup(L1_edge_in_pong_buffer_name)._type.referencedType = ctxt.lookup(
            operatorRepresentation['data_in_net'])._type.referencedType
      ctxt.lookup(L1_edge_out_ping_buffer_name)._type.referencedType = ctxt.lookup(
            operatorRepresentation['data_out'])._type.referencedType
      ctxt.lookup(L1_edge_out_pong_buffer_name)._type.referencedType = ctxt.lookup(
            operatorRepresentation['data_out'])._type.referencedType
      
      ctxt.lookup(L1_edge_in_ping_buffer_name)._targetmemorylevel['CustomnColSum'] = "L1"
      ctxt.lookup(L1_edge_in_pong_buffer_name)._targetmemorylevel['CustomnColSum'] = "L1"
      ctxt.lookup(L1_edge_out_ping_buffer_name)._targetmemorylevel['CustomnColSum'] = "L1"
      ctxt.lookup(L1_edge_out_pong_buffer_name)._targetmemorylevel['CustomnColSum'] = "L1"
      
      operatorRepresentation['edge_buff_l1_ping_in']    = L1_edge_in_ping_buffer_name
      operatorRepresentation['edge_buff_l1_ping_out']   = L1_edge_out_ping_buffer_name
      operatorRepresentation['edge_buff_l1_pong_in']    = L1_edge_in_pong_buffer_name
      operatorRepresentation['edge_buff_l1_pong_out']   = L1_edge_out_pong_buffer_name

      return ctxt, operatorRepresentation, [L1_edge_in_ping_buffer_name, L1_edge_out_ping_buffer_name, 
                                            L1_edge_in_pong_buffer_name, L1_edge_out_pong_buffer_name]


referenceTemplate = CustomnColSum("""
// Customized columnwise sum for DPVO (Name: ${nodeName}, Op: ${nodeOp})

printf("start col sum kernel\\n");                            
ColSum_master_kernel( (float *)${data_in_net}, \
                          (int   *)${data_in_kk}, \
                          (float *)${data_out}, \
                          (float *)${edge_buff_l1_ping_in}, \
                          (float *)${edge_buff_l1_ping_out}, \
                          (float *)${edge_buff_l1_pong_in}, \
                          (float *)${edge_buff_l1_pong_out}, \
                          (int    )${dir});

    
                              
""")