# SPDX-FileCopyrightText: 2025 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0

from typing import Dict, List, Tuple

import numpy as np

from Deeploy.DeeployTypes import NetworkContext, NodeTemplate, OperatorRepresentation


class NE16GEMMTemplate(NodeTemplate):

    def __init__(self, templateStr):
        super().__init__(templateStr)

    def alignToContext(self, ctxt: NetworkContext,
                       operatorRepresentation: OperatorRepresentation) -> Tuple[NetworkContext, Dict, List[str]]:

        A = ctxt.lookup(operatorRepresentation['A'])
        data_out = ctxt.lookup(operatorRepresentation['data_out'])

        input_signed = A._type.referencedType.typeMin < 0
        output_bits = data_out._type.referencedType.typeWidth
        output_signed = data_out._type.referencedType.typeMin < 0

        operatorRepresentation['input_signed'] = input_signed
        operatorRepresentation['output_bits'] = output_bits
        operatorRepresentation['quant_bits'] = 2 if output_bits == 32 else 0
        operatorRepresentation['quant_norect'] = 1 if (output_bits == 32 or output_signed) else 0

        return ctxt, operatorRepresentation, []


# 8-bit output template (RequantizedGemm) — uses tiled ${mul} and ${scale_n}
referenceTemplate = NE16GEMMTemplate("""
// NE16 Linear 8-bit (Name: ${nodeName}, Op: ${nodeOp})

% if input_signed:
// Signed input: add 128 offset to convert int8 -> uint8 (multi-core SIMD)
{
    ne16_int8_to_uint8_T _offset_arg = {
        .In = (int8_t *)${A},
        .Out = (uint8_t *)${A},
        .size = ${batch} * ${M} * ${N}
    };
    pi_cl_team_fork(NUM_CORES, (void *)ne16_int8_to_uint8, &_offset_arg);
}
% endif

{
    unsigned int _ne16_cfg = 0;
    _ne16_cfg |= ((8 - 1)              & NE16_MASK_WBITS_M1)         << NE16_SHIFT_WBITS_M1;
    _ne16_cfg |= (0                     & NE16_MASK_MODE16)            << NE16_SHIFT_MODE16;
    _ne16_cfg |= (1                     & NE16_MASK_OUTQUANT)          << NE16_SHIFT_OUTQUANT;
    _ne16_cfg |= (NE16_FILTER_MODE_1x1  & NE16_MASK_FILTER_MODE)       << NE16_SHIFT_FILTER_MODE;
    _ne16_cfg |= (0                     & NE16_MASK_LINEAR_MODE)       << NE16_SHIFT_LINEAR_MODE;
    _ne16_cfg |= (0                     & NE16_MASK_STRIDED_MODE)      << NE16_SHIFT_STRIDED_MODE;
    _ne16_cfg |= (NE16_BITS_8BIT        & NE16_MASK_NORM_BITS)         << NE16_SHIFT_NORM_BITS;
    _ne16_cfg |= (0                     & NE16_MASK_STREAMIN)          << NE16_SHIFT_STREAMIN;
    _ne16_cfg |= (1                     & NE16_MASK_WEIGHT_OFFSET_CFG) << NE16_SHIFT_WEIGHT_OFFSET_CFG;
    _ne16_cfg |= (0                     & NE16_MASK_QUANT_RIGHT_SHIFT) << NE16_SHIFT_QUANT_RIGHT_SHIFT;
    _ne16_cfg |= (${quant_bits}         & NE16_MASK_QUANT_BITS)        << NE16_SHIFT_QUANT_BITS;
    _ne16_cfg |= (${quant_norect}       & NE16_MASK_QUANT_NORECT)      << NE16_SHIFT_QUANT_NORECT;
    _ne16_cfg |= (1                     & NE16_MASK_NORM_SHIFT)        << NE16_SHIFT_NORM_SHIFT;
    _ne16_cfg |= (1                     & NE16_MASK_NORM_BIAS)         << NE16_SHIFT_NORM_BIAS;

    NE16_Enable();
    NE16_SoftReset();

    KerConv_NE16_T _ne16_arg = {
        .In                   = (void *)${A},
        .Filter               = (unsigned short *)${B},
        .Bias                 = (int *)${C},
        .Out                  = (void *)${data_out},
        .Scale                = (unsigned char *)${mul},
        .ScaleN               = (unsigned char *)${scale_n},
        .Tile_InFeat          = ${N},
        .TotalInFeatures      = ${N},
        .Tile_InH             = 1,
        .Tile_InW             = ${batch} * ${M},
        .Tile_OutFeat         = ${O},
        .Tile_OutH            = 1,
        .Tile_OutW            = ${batch} * ${M},
        .FilterSize           = 1,
        .Pad_Val              = 0,
        .Pad                  = (v4s){0, 0, 0, 0},
        .W_Offset             = -128,
        .Qw                   = 8,
        .Mode16               = 0,
        .FirstD0              = 1,
        .LastD0               = 1,
        .Default_NE16_Job_Cfg = _ne16_cfg,
        .Fx                   = 1,
        .Fy                   = 1,
        .Sx                   = 1,
        .Sy                   = 1,
        .Dx                   = 1,
        .Dy                   = 1,
        .BuffOut              = NULL,
        .Infos                = NULL,
        .Extra                = NULL,
    };
    KerConv1x1_SmallHW_Stride1_NE16(&_ne16_arg);

    NE16_Disable();
}
""")

# Int32 output template (plain Gemm) — hardcoded scale=1, scale_n=0
int32OutputTemplate = NE16GEMMTemplate("""
// NE16 Linear Int32 (Name: ${nodeName}, Op: ${nodeOp})

% if input_signed:
// Signed input: add 128 offset to convert int8 -> uint8 (multi-core SIMD)
{
    ne16_int8_to_uint8_T _offset_arg = {
        .In = (int8_t *)${A},
        .Out = (uint8_t *)${A},
        .size = ${batch} * ${M} * ${N}
    };
    pi_cl_team_fork(NUM_CORES, (void *)ne16_int8_to_uint8, &_offset_arg);
}
% endif

{
    unsigned char _ne16_ones[${O}];
    unsigned char _ne16_zeros[${O}];
    memset(_ne16_ones, 1, ${O});
    memset(_ne16_zeros, 0, ${O});

    unsigned int _ne16_cfg = 0;
    _ne16_cfg |= ((8 - 1)              & NE16_MASK_WBITS_M1)         << NE16_SHIFT_WBITS_M1;
    _ne16_cfg |= (0                     & NE16_MASK_MODE16)            << NE16_SHIFT_MODE16;
    _ne16_cfg |= (1                     & NE16_MASK_OUTQUANT)          << NE16_SHIFT_OUTQUANT;
    _ne16_cfg |= (NE16_FILTER_MODE_1x1  & NE16_MASK_FILTER_MODE)       << NE16_SHIFT_FILTER_MODE;
    _ne16_cfg |= (0                     & NE16_MASK_LINEAR_MODE)       << NE16_SHIFT_LINEAR_MODE;
    _ne16_cfg |= (0                     & NE16_MASK_STRIDED_MODE)      << NE16_SHIFT_STRIDED_MODE;
    _ne16_cfg |= (NE16_BITS_8BIT        & NE16_MASK_NORM_BITS)         << NE16_SHIFT_NORM_BITS;
    _ne16_cfg |= (0                     & NE16_MASK_STREAMIN)          << NE16_SHIFT_STREAMIN;
    _ne16_cfg |= (1                     & NE16_MASK_WEIGHT_OFFSET_CFG) << NE16_SHIFT_WEIGHT_OFFSET_CFG;
    _ne16_cfg |= (0                     & NE16_MASK_QUANT_RIGHT_SHIFT) << NE16_SHIFT_QUANT_RIGHT_SHIFT;
    _ne16_cfg |= (${quant_bits}         & NE16_MASK_QUANT_BITS)        << NE16_SHIFT_QUANT_BITS;
    _ne16_cfg |= (${quant_norect}       & NE16_MASK_QUANT_NORECT)      << NE16_SHIFT_QUANT_NORECT;
    _ne16_cfg |= (1                     & NE16_MASK_NORM_SHIFT)        << NE16_SHIFT_NORM_SHIFT;
    _ne16_cfg |= (1                     & NE16_MASK_NORM_BIAS)         << NE16_SHIFT_NORM_BIAS;

    NE16_Enable();
    NE16_SoftReset();

    KerConv_NE16_T _ne16_arg = {
        .In                   = (void *)${A},
        .Filter               = (unsigned short *)${B},
        .Bias                 = (int *)${C},
        .Out                  = (void *)${data_out},
        .Scale                = _ne16_ones,
        .ScaleN               = _ne16_zeros,
        .Tile_InFeat          = ${N},
        .TotalInFeatures      = ${N},
        .Tile_InH             = 1,
        .Tile_InW             = ${batch} * ${M},
        .Tile_OutFeat         = ${O},
        .Tile_OutH            = 1,
        .Tile_OutW            = ${batch} * ${M},
        .FilterSize           = 1,
        .Pad_Val              = 0,
        .Pad                  = (v4s){0, 0, 0, 0},
        .W_Offset             = -128,
        .Qw                   = 8,
        .Mode16               = 0,
        .FirstD0              = 1,
        .LastD0               = 1,
        .Default_NE16_Job_Cfg = _ne16_cfg,
        .Fx                   = 1,
        .Fy                   = 1,
        .Sx                   = 1,
        .Sy                   = 1,
        .Dx                   = 1,
        .Dy                   = 1,
        .BuffOut              = NULL,
        .Infos                = NULL,
        .Extra                = NULL,
    };
    KerConv1x1_SmallHW_Stride1_NE16(&_ne16_arg);

    NE16_Disable();
}
""")
