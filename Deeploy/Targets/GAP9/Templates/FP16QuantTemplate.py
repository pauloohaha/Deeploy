# SPDX-FileCopyrightText: 2025 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0
#
# FP16 Quant/Dequant templates for GAP9.
# int8/uint8 variants use GAP SDK kernels (CNN_Copy.c).
# int32 variants use custom kernels (quant_fp16.c).
# All called via GAP9Transformer which handles pi_cl_team_fork.

from Deeploy.DeeployTypes import NodeTemplate

# === Dequant templates: int → fp16 ===

# int8 → fp16: SDK kernel CNN_FpsIEEE16
fp16DequantI8Template = NodeTemplate("""
// FP16 Dequant int8→fp16 (Name: ${nodeName}, Op: ${nodeOp})
{
    signed char _dq_infos[8];
    *((float *)(_dq_infos + 0)) = (float)(-(${zero_point}));
    *((float *)(_dq_infos + 4)) = (float)(${scale});
    CNN_Quantize_T _dq_arg = {
        .In = (void *)${data_in},
        .Out = (void *)${data_out},
        .W = ${size},
        .H = 1,
        .Infos = _dq_infos,
    };
    CNN_FpsIEEE16(&_dq_arg);
}
""")

# uint8 → fp16: SDK kernel CNN_UFpsIEEE16
fp16DequantU8Template = NodeTemplate("""
// FP16 Dequant uint8→fp16 (Name: ${nodeName}, Op: ${nodeOp})
{
    signed char _dq_infos[8];
    *((float *)(_dq_infos + 0)) = (float)(-(${zero_point}));
    *((float *)(_dq_infos + 4)) = (float)(${scale});
    CNN_Quantize_T _dq_arg = {
        .In = (void *)${data_in},
        .Out = (void *)${data_out},
        .W = ${size},
        .H = 1,
        .Infos = _dq_infos,
    };
    CNN_UFpsIEEE16(&_dq_arg);
}
""")

# === Quant templates: fp16 → int ===

# fp16 → int8: SDK kernel CNN_IEEE16Fps
fp16QuantI8Template = NodeTemplate("""
// FP16 Quant fp16→int8 (Name: ${nodeName}, Op: ${nodeOp})
{
    signed char _q_infos[8];
    *((float *)(_q_infos + 0)) = (float)(${zero_point});
    *((float *)(_q_infos + 4)) = (float)(${scale});
    CNN_Quantize_T _q_arg = {
        .In = (void *)${data_in},
        .Out = (void *)${data_out},
        .W = ${size},
        .H = 1,
        .Infos = _q_infos,
    };
    CNN_IEEE16Fps(&_q_arg);
}
""")

# fp16 → uint8: SDK kernel CNN_IEEE16UFps
fp16QuantU8Template = NodeTemplate("""
// FP16 Quant fp16→uint8 (Name: ${nodeName}, Op: ${nodeOp})
{
    signed char _q_infos[8];
    *((float *)(_q_infos + 0)) = (float)(${zero_point});
    *((float *)(_q_infos + 4)) = (float)(${scale});
    CNN_Quantize_T _q_arg = {
        .In = (void *)${data_in},
        .Out = (void *)${data_out},
        .W = ${size},
        .H = 1,
        .Infos = _q_infos,
    };
    CNN_IEEE16UFps(&_q_arg);
}
""")

