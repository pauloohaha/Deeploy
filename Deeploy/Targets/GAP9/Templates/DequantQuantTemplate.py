# SPDX-FileCopyrightText: 2025 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0

from Deeploy.DeeployTypes import NodeTemplate

referenceTemplate = NodeTemplate("""
// Dequant -> Quant in fp32 (Name: ${nodeName}, Op: ${nodeOp})
GAP9_DequantQuant_${'s' if data_in_type.referencedType.typeMin < 0 else 'u'}${data_in_type.referencedType.typeWidth}_${'s' if data_out_type.referencedType.typeMin < 0 else 'u'}${data_out_type.referencedType.typeWidth}(${data_in}, ${data_out}, ${size}, (float)(-(${in_zero_point})), (float)(${in_scale}), (float)(${out_zero_point}), (float)(${out_scale}));
""")
