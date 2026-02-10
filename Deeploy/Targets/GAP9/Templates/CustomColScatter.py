# SPDX-FileCopyrightText: 2025 ETH Zurich and University of Bologna
# Author: Pu Deng, ETH Zurich
# SPDX-License-Identifier: Apache-2.0

from typing import Dict, List, Tuple

from Deeploy.DeeployTypes import NetworkContext, NodeTemplate, OperatorRepresentation, VariableBuffer
from Deeploy.Targets.GAP9.Templates.DPVO_defines import MAX_PATCH_PER_FRAME, MAX_EDGE_PER_PATCH, DIM

class CustomColScatter(NodeTemplate):
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
      in_agg_Buffer  = ctxt.lookup(operatorRepresentation['data_in_agg'])
      out_net_Buffer = ctxt.lookup(operatorRepresentation['data_out'])
      out_net_Buffer._alias = in_net_Buffer.name

      #expect the input and output buffers in L2
      in_net_Buffer._targetmemorylevel['CustomColScatter'] = 'L2'
      in_kk_Buffer._targetmemorylevel['CustomColScatter'] = 'L2'
      in_agg_Buffer._targetmemorylevel['CustomColScatter'] = 'L2'
      out_net_Buffer._targetmemorylevel['CustomColScatter'] = 'L2'

      return ctxt, operatorRepresentation, []
    
    
    def hoistTransientBuffers(self, ctxt: NetworkContext,
                              operatorRepresentation: OperatorRepresentation) -> Tuple[NetworkContext, Dict, List[str]]:

      L1_edge_ping_buffer_name          = operatorRepresentation['nodeName'] + "_L1_edge_ping_buffer"
      L1_edge_pong_buffer_name          = operatorRepresentation['nodeName'] + "_L1_edge_pong_buffer"
      L1_agg_ping_buffer_name           = operatorRepresentation['nodeName'] + "_L1_agg_ping_buffer"
      L1_agg_pong_buffer_name           = operatorRepresentation['nodeName'] + "_L1_agg_pong_buffer"

      if operatorRepresentation['dir'] == 0:
          # patch aggregation, process MAX_EDGE_PER_PATCH from one patch each time
          L1_edge_buffer_dim = MAX_EDGE_PER_PATCH * DIM * 4 #float32
      else:
          # frame aggregation, process MAX_PATCH_PER_FRAME from one frame each time
          L1_edge_buffer_dim = MAX_PATCH_PER_FRAME * DIM * 4 #float32
          
      L1_agg_buffer_size = DIM * 4 #int32

      ctxt.hoistTransientBuffer(L1_edge_ping_buffer_name, L1_edge_buffer_dim)
      ctxt.hoistTransientBuffer(L1_edge_pong_buffer_name, L1_edge_buffer_dim)
      ctxt.hoistTransientBuffer(L1_agg_ping_buffer_name, L1_agg_buffer_size)
      ctxt.hoistTransientBuffer(L1_agg_pong_buffer_name, L1_agg_buffer_size)
      
      ctxt.lookup(L1_edge_ping_buffer_name)._type.referencedType = ctxt.lookup(
            operatorRepresentation['data_in_net'])._type.referencedType
      ctxt.lookup(L1_edge_pong_buffer_name)._type.referencedType = ctxt.lookup(
            operatorRepresentation['data_in_net'])._type.referencedType
      ctxt.lookup(L1_agg_ping_buffer_name)._type.referencedType = ctxt.lookup(
            operatorRepresentation['data_in_net'])._type.referencedType
      ctxt.lookup(L1_agg_pong_buffer_name)._type.referencedType = ctxt.lookup(
            operatorRepresentation['data_in_net'])._type.referencedType
      
      ctxt.lookup(L1_edge_ping_buffer_name)._targetmemorylevel['CustomColScatter'] = "L1"
      ctxt.lookup(L1_edge_pong_buffer_name)._targetmemorylevel['CustomColScatter'] = "L1"
      ctxt.lookup(L1_agg_ping_buffer_name)._targetmemorylevel['CustomColScatter'] = "L1"
      ctxt.lookup(L1_agg_pong_buffer_name)._targetmemorylevel['CustomColScatter'] = "L1"
    
      
      operatorRepresentation['edge_buff_l1_ping']   = L1_edge_ping_buffer_name
      operatorRepresentation['edge_buff_l1_pong']   = L1_edge_pong_buffer_name
      operatorRepresentation['agg_buff_ping']       = L1_agg_ping_buffer_name
      operatorRepresentation['agg_buff_pong']       = L1_agg_pong_buffer_name

      return ctxt, operatorRepresentation, [L1_edge_ping_buffer_name, L1_edge_pong_buffer_name, 
                                            L1_agg_ping_buffer_name, L1_agg_pong_buffer_name]


referenceTemplate = CustomColScatter("""
// Customized columnwise scatter add for DPVO (Name: ${nodeName}, Op: ${nodeOp})
                 
ColScatter_master_kernel( (float *)${data_in_net}, \
                          (int   *)${data_in_kk}, \
                          (float *)${data_in_agg}, \
                          (float *)${data_out}, \
                          (float *)${edge_buff_l1_ping}, \
                          (float *)${agg_buff_ping}, \
                          (float *)${edge_buff_l1_pong}, \
                          (float *)${agg_buff_pong},    \
                          (int    )${dir});

    
                              
""")