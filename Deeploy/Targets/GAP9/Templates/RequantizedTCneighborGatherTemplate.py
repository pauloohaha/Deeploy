# SPDX-FileCopyrightText: 2025 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0
#
# RequantShift -> TCneighborGather fused: the requantization (same arithmetic and parameters as the PULP
# UniformRequantShift template: no input/output offset, rounding on) is applied while gathering.

from Deeploy.DeeployTypes import NodeTemplate

referenceTemplate = NodeTemplate("""
// TC neighbor gather with fused requantization (Name: ${nodeName}, Op: ${nodeOp})
TC_layout_neighbor_gather_rqs_int8_T gather_arg;
gather_arg.In            = (int8_t *)${data_in_net};
gather_arg.Out           = (int8_t *)${data_out};
gather_arg.kk_buff       = (int *)${data_in_kk};
gather_arg.dir           = ${dir};
gather_arg.dim           = ${dim};
gather_arg.mul           = ${rqs_mul};
gather_arg.add           = ${rqs_add};
gather_arg.log2D         = ${log2D};
gather_arg.input_offset  = 0;
gather_arg.output_offset = 0;
gather_arg.output_min    = ${output_min};
gather_arg.output_max    = ${output_max};
gather_arg.rounding      = 1;
TC_layout_neighbor_gather_rqs_int8(&gather_arg);
""")
