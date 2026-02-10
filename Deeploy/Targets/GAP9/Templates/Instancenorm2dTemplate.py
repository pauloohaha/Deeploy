# SPDX-FileCopyrightText: 2023 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0

from Deeploy.DeeployTypes import NodeTemplate

referenceTemplate = NodeTemplate("""
// Float Instancenorm2d (Name: ${nodeName}, Op: ${nodeOp})
PULP_Instancenorm2d_fp32_fp32(
    ${data_in},
    ${data_out},
    ${weight},
    ${bias},
    ${size},
    ${last2DimLength},
    ${epsilon}
);
""")