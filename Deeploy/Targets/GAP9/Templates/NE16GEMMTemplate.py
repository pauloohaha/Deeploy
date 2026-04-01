# SPDX-FileCopyrightText: 2025 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0

import math
from typing import Dict, List, Tuple

import numpy as np

from Deeploy.AbstractDataTypes import PointerClass
from Deeploy.CommonExtensions.DataTypes import uint8_t
from Deeploy.DeeployTypes import ConstantBuffer, NetworkContext, NodeTemplate, OperatorRepresentation


def _ne16_conv_1x1_weight_layout(W, w_bits=8):
    """Pack int8 weights for NE16 1x1 conv mode.
    W: int8 [Ko, Ki] -> uint8 [Ko, Nb_KI, Qw, 2] (bitplane packed)
    Weights stored as uint8 = int8 + 128.
    """
    tp_in = 16
    Ko_, Ki_ = W.shape
    W_uint8 = (W.astype(np.int32) + 128).astype(np.uint8)
    nb_ki = (Ki_ + tp_in - 1) // tp_in
    w_binary = np.zeros((Ko_ * nb_ki, w_bits, 8, tp_in // 8), dtype=np.uint8)
    for ko in range(Ko_):
        for ki_maj in range(nb_ki):
            for ki_min in range(tp_in):
                idx = ko * nb_ki + ki_maj
                ki = ki_maj * tp_in + ki_min
                val = int(W_uint8[ko, ki]) if ki < Ki_ else 0
                for q in range(w_bits):
                    w_binary[idx, q, ki_min % 8, ki_min // 8] = (val >> q) & 1
    space = np.logspace(0, 7, num=8, base=2, dtype=np.int32).reshape((8, 1))
    w_layout = np.sum(w_binary * space, axis=2, dtype=np.uint8)
    return w_layout.reshape((Ko_, nb_ki, w_bits, tp_in // 8))


def _compute_ne16_scale_shift(mul_values, log2D):
    """Convert Deeploy's mul/log2D to NE16's per-channel scale/scale_n.
    scale_factor = mul[ko] / 2^log2D
    Decompose: scale_factor ≈ scale[ko] * 2^(-scale_n[ko])
    """
    Ko = len(mul_values)
    ne16_scale = np.zeros(Ko, dtype=np.uint8)
    ne16_scale_n = np.zeros(Ko, dtype=np.uint8)

    for ko in range(Ko):
        sf = float(mul_values[ko]) / float(2 ** log2D)
        if sf >= 1.0:
            sn = 0
            sc = min(255, max(1, int(round(sf))))
        elif sf > 0:
            sn = min(31, max(0, int(math.floor(math.log2(127.0 / sf)))))
            sc = min(255, max(1, int(round(sf * (1 << sn)))))
        else:
            sn = 0
            sc = 0
        ne16_scale[ko] = sc
        ne16_scale_n[ko] = sn

    return ne16_scale, ne16_scale_n


def _build_ne16_cfg(signed_output=False, output_bits=8):
    """Build NE16 config word for 1x1 conv. output_bits=8 for int8/uint8, output_bits=32 for int32."""
    NE16_SHIFT_WBITS_M1 = 0
    NE16_SHIFT_MODE16 = 3
    NE16_SHIFT_OUTQUANT = 4
    NE16_SHIFT_FILTER_MODE = 5
    NE16_SHIFT_LINEAR_MODE = 7
    NE16_SHIFT_STRIDED_MODE = 8
    NE16_SHIFT_NORM_BITS = 9
    NE16_SHIFT_STREAMIN = 11
    NE16_SHIFT_WEIGHT_OFFSET_CFG = 12
    NE16_SHIFT_QUANT_RIGHT_SHIFT = 13
    NE16_SHIFT_QUANT_BITS = 21
    NE16_SHIFT_QUANT_NORECT = 23
    NE16_SHIFT_NORM_SHIFT = 24
    NE16_SHIFT_NORM_BIAS = 25

    NE16_FILTER_MODE_1x1 = 3

    cfg = 0
    cfg |= (8 - 1) << NE16_SHIFT_WBITS_M1  # 8-bit weights
    cfg |= 0 << NE16_SHIFT_MODE16  # 8-bit mode
    cfg |= 1 << NE16_SHIFT_OUTQUANT  # enable normquant
    cfg |= NE16_FILTER_MODE_1x1 << NE16_SHIFT_FILTER_MODE
    cfg |= 0 << NE16_SHIFT_LINEAR_MODE
    cfg |= 0 << NE16_SHIFT_STRIDED_MODE
    cfg |= 0 << NE16_SHIFT_NORM_BITS  # 8-bit scale
    cfg |= 0 << NE16_SHIFT_STREAMIN
    cfg |= 1 << NE16_SHIFT_WEIGHT_OFFSET_CFG  # per-layer weight offset
    cfg |= 0 << NE16_SHIFT_QUANT_RIGHT_SHIFT
    cfg |= (2 if output_bits == 32 else 0) << NE16_SHIFT_QUANT_BITS  # 2=int32 output, 0=8-bit output
    cfg |= (1 if (output_bits == 32 or signed_output) else 0) << NE16_SHIFT_QUANT_NORECT  # 1=no clip/signed, 0=unsigned [0,255]
    cfg |= 1 << NE16_SHIFT_NORM_SHIFT  # per-channel shift
    cfg |= 1 << NE16_SHIFT_NORM_BIAS  # per-channel bias
    return cfg


class NE16GEMMTemplate(NodeTemplate):

    def __init__(self, templateStr):
        super().__init__(templateStr)

    def alignToContext(self, ctxt: NetworkContext,
                       operatorRepresentation: OperatorRepresentation) -> Tuple[NetworkContext, Dict, List[str]]:

        A = ctxt.lookup(operatorRepresentation['A'])
        B = ctxt.lookup(operatorRepresentation['B'])
        data_out = ctxt.lookup(operatorRepresentation['data_out'])

        input_signed = A._type.referencedType.typeMin < 0
        operatorRepresentation['input_signed'] = input_signed

        output_bits = data_out._type.referencedType.typeWidth
        output_signed = data_out._type.referencedType.typeMin < 0

        weight_buf = ctxt.lookup(operatorRepresentation['B'])
        assert isinstance(weight_buf, ConstantBuffer), "NE16 GEMM requires constant weights"

        w_int8 = weight_buf.values.astype(np.int8)
        Ko = int(operatorRepresentation['O'])
        Ki = int(operatorRepresentation['N'])

        # Pack weights in NE16 bitplane format
        ne16_weights = _ne16_conv_1x1_weight_layout(w_int8.reshape(Ko, Ki))
        weight_buf.values = ne16_weights.flatten()

        add_buf = ctxt.lookup(operatorRepresentation['C'])
        assert isinstance(add_buf, ConstantBuffer), "NE16 GEMM requires constant add/bias"
        raw_add = add_buf.values.flatten().astype(np.int64)
        # GEMMLayer.computeShapes broadcasts bias from [Ko] to [M, Ko] — extract per-channel
        if raw_add.size > Ko:
            add_values = raw_add.reshape(-1, Ko)[0]
        else:
            add_values = raw_add

        # Signed input bias compensation
        if input_signed:
            w_sum = w_int8.reshape(Ko, Ki).astype(np.int64).sum(axis=1)
            add_values = add_values - 128 * w_sum

        if output_bits == 32:
            # Int32 output: scale=1, scale_n=0, bias unchanged (no requant)
            ne16_scale = np.ones(Ko, dtype=np.uint8)
            ne16_scale_n = np.zeros(Ko, dtype=np.uint8)
            ne16_bias = add_values.astype(np.int32)
        else:
            # 8-bit output: compute scale/scale_n from mul/log2D
            mul_buf = ctxt.lookup(operatorRepresentation['mul'])
            assert isinstance(mul_buf, ConstantBuffer), "NE16 GEMM requires constant mul"
            mul_values = mul_buf.values.flatten().astype(np.int32)
            log2D = operatorRepresentation['log2D']

            ne16_scale, ne16_scale_n = _compute_ne16_scale_shift(mul_values, log2D)
            ne16_bias = (add_values * ne16_scale.astype(np.int64)).astype(np.int32)

            # Update mul buffer with NE16 scale
            mul_buf.values = ne16_scale

        # Update bias buffer
        add_buf.values = ne16_bias

        # Store scale_n as a new constant buffer
        scale_n_name = operatorRepresentation['nodeName'] + "_scale_n"
        scale_n_buf = ctxt.ConstantBuffer(scale_n_name, [Ko], ne16_scale_n)
        scale_n_buf._type = PointerClass(uint8_t)
        ctxt.add(scale_n_buf, 'global')
        operatorRepresentation['scale_n'] = scale_n_name

        if output_bits == 32:
            # For int32 output, create scale buffer and map to template vars
            scale_name = operatorRepresentation['nodeName'] + "_ne16_scale"
            scale_buf = ctxt.ConstantBuffer(scale_name, [Ko], ne16_scale)
            scale_buf._type = PointerClass(uint8_t)
            ctxt.add(scale_buf, 'global')
            operatorRepresentation['mul'] = scale_name

        # NE16 config
        operatorRepresentation['ne16_cfg'] = _build_ne16_cfg(signed_output=output_signed, output_bits=output_bits)

        return ctxt, operatorRepresentation, []


referenceTemplate = NE16GEMMTemplate("""
// NE16 Linear (Name: ${nodeName}, Op: ${nodeOp})

% if input_signed:
// Signed input: add 128 offset to convert int8 -> uint8
{
    uint8_t *_ne16_in = (uint8_t *)${A};
    int _ne16_size = ${batch} * ${M} * ${N};
    for (int _i = 0; _i < _ne16_size; _i++) {
        _ne16_in[_i] = (uint8_t)((int32_t)((int8_t *)${A})[_i] + 128);
    }
}
% endif

NE16_Enable();
NE16_SoftReset();

{
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
        .Default_NE16_Job_Cfg = ${ne16_cfg},
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
}

NE16_Disable();
""")
