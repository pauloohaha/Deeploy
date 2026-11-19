# SPDX-FileCopyrightText: 2025 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0
#
# Add -> RequantShift fused (same arithmetic and parameters as the Generic Add followed by the PULP
# UniformRequantShift template: no input/output offset, rounding on), run on all cores.

from Deeploy.DeeployTypes import NodeTemplate

referenceTemplate = NodeTemplate("""
// Add with fused requantization (Name: ${nodeName}, Op: ${nodeOp})
GAP9_AddRQS_s8_s8_s8(${data_in_1}, ${data_in_2}, ${data_out}, ${size}, ${rqs_mul}, ${rqs_add}, ${log2D}, 0, 0, ${output_min}, ${output_max}, 1);
""")
