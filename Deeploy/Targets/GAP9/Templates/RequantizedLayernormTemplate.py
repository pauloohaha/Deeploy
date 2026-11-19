# SPDX-FileCopyrightText: 2025 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0
#
# Dequant -> LayerNorm (fp32) -> Quant fused into one kernel (TargetLibraries/GAP9/src/RequantizedLayernorm.c).
# The constants are formatted exactly like the SDK Dequant/Quant templates (GAP9SDKDequantQuantTemplate), so the
# fused kernel sees the same float values as the unfused chain.

from Deeploy.DeeployTypes import NodeTemplate

referenceTemplate = NodeTemplate("""
// Requantized Layernorm (Name: ${nodeName}, Op: ${nodeOp})
GAP9_RQLayernorm_${'s' if data_in_type.referencedType.typeMin < 0 else 'u'}${data_in_type.referencedType.typeWidth}_${'s' if data_out_type.referencedType.typeMin < 0 else 'u'}${data_out_type.referencedType.typeWidth}(
    ${data_in},
    ${data_out},
    ${weight},
    ${bias},
    ${size},
    ${lastDimLength},
    ${epsilon},
    (float)(-(${in_zero_point})),
    (float)(${in_scale}),
    (float)(${out_zero_point}),
    (float)(${out_scale})
);
""")
