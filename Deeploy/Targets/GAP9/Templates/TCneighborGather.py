# SPDX-FileCopyrightText: 2023 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0

from Deeploy.DeeployTypes import NodeTemplate

referenceTemplate = NodeTemplate("""
// TC neighbor gather (Name: ${nodeName}, Op: ${nodeOp})
                                 
TC_layout_neighbor_gather_T gather_arg;
gather_arg.In       = (float  *)${data_in_net};
gather_arg.Out      = (float  *)${data_out};
gather_arg.kk_buff  = (int    *)${data_in_kk};
gather_arg.dir      = ${dir};
                                 
TC_layout_neighbor_gather(&gather_arg);
""")