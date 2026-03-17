# SPDX-FileCopyrightText: 2023 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0

from Deeploy.DeeployTypes import NodeTemplate

referenceTemplate = NodeTemplate("""
// TC neighbor gather (Name: ${nodeName}, Op: ${nodeOp})
                                 
TC_layout_neighbor_gather_int8_T gather_arg;
gather_arg.In       = (int8_t  *)${data_in_net};
gather_arg.Out      = (int8_t  *)${data_out};
gather_arg.kk_buff  = (int    *)${data_in_kk};
gather_arg.dir      = ${dir};
                                 
TC_layout_neighbor_gather_int8(&gather_arg);
""")