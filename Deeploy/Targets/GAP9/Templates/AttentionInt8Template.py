# SPDX-FileCopyrightText: 2025 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0
#
# Fused integer-IO softmax aggregation / scatter kernels (TargetLibraries/GAP9/src/AttentionInt8.c), created by
# GAP9AttentionInt8FusionPass. Like the CustomCol* templates, the operands stay whole in L2 and the kernel moves
# its own tiles through hoisted L1 buffers.

from typing import Dict, List, Tuple

from Deeploy.CommonExtensions.DataTypes import float32_t, int8_t, int16_t
from Deeploy.DeeployTypes import NetworkContext, NodeTemplate, OperatorRepresentation
from Deeploy.Targets.GAP9.Templates.DPVO_defines import DIM, MAX_EDGE_PER_PATCH, MAX_PATCH_PER_FRAME

_MAX_ROWS = max(MAX_EDGE_PER_PATCH, MAX_PATCH_PER_FRAME)


class _L2OperandsTemplate(NodeTemplate):
    """Operands in L2, transient L1 buffers given by `l1Buffers` = [(suffix, bytes, type), ...]."""

    operandKeys: List[str] = []
    l1Buffers: List[Tuple[str, int, type]] = []

    def alignToContext(self, ctxt: NetworkContext,
                       operatorRepresentation: OperatorRepresentation) -> Tuple[NetworkContext, Dict, List[str]]:
        op = operatorRepresentation['nodeOp']
        for key in self.operandKeys:
            ctxt.lookup(operatorRepresentation[key])._targetmemorylevel[op] = 'L2'
        return ctxt, operatorRepresentation, []

    def hoistTransientBuffers(self, ctxt: NetworkContext,
                              operatorRepresentation: OperatorRepresentation) -> Tuple[NetworkContext, Dict, List[str]]:
        names = []
        for suffix, size, refType in self.l1Buffers:
            name = operatorRepresentation['nodeName'] + "_L1_" + suffix
            ctxt.hoistTransientBuffer(name, size)
            ctxt.lookup(name)._type.referencedType = refType
            operatorRepresentation['l1_' + suffix] = name
            names.append(name)
        return ctxt, operatorRepresentation, names


class _ColAttnPoolTemplate(_L2OperandsTemplate):
    operandKeys = ['data_in_g', 'data_in_f', 'data_in_kk', 'data_out']
    l1Buffers = [('g_ping', _MAX_ROWS * DIM, int8_t), ('g_pong', _MAX_ROWS * DIM, int8_t),
                 ('f_ping', _MAX_ROWS * DIM, int8_t), ('f_pong', _MAX_ROWS * DIM, int8_t),
                 ('out', _MAX_ROWS * DIM, int8_t)]


class _ScatterQuantTemplate(_L2OperandsTemplate):
    operandKeys = ['data_in_net', 'data_in_agg', 'data_in_kk', 'data_out']
    l1Buffers = [('net_ping', MAX_PATCH_PER_FRAME * DIM * 2, int16_t),
                 ('net_pong', MAX_PATCH_PER_FRAME * DIM * 2, int16_t), ('out_ping', MAX_PATCH_PER_FRAME * DIM, int8_t),
                 ('out_pong', MAX_PATCH_PER_FRAME * DIM, int8_t), ('agg', _MAX_ROWS * DIM, int8_t)]


class _ScatterScatterRQLNTemplate(_L2OperandsTemplate):
    operandKeys = ['data_in_net', 'data_in_agg1', 'data_in_agg2', 'data_in_kk', 'weight', 'bias', 'data_out']
    l1Buffers = [('net_ping', MAX_PATCH_PER_FRAME * DIM * 2, int16_t),
                 ('net_pong', MAX_PATCH_PER_FRAME * DIM * 2, int16_t), ('out_ping', MAX_PATCH_PER_FRAME * DIM, int8_t),
                 ('out_pong', MAX_PATCH_PER_FRAME * DIM, int8_t), ('agg1', _MAX_ROWS * DIM, int8_t),
                 ('agg2', _MAX_ROWS * DIM, int8_t), ('weight', DIM * 4, float32_t), ('bias', DIM * 4, float32_t)]


colAttnPoolTemplate = _ColAttnPoolTemplate("""
// Fused softmax aggregation, int8 in / int8 out (Name: ${nodeName}, Op: ${nodeOp})
GAP9_ColAttnPool_s8((int8_t *)${data_in_g}, (int8_t *)${data_in_f}, (int32_t *)${data_in_kk}, (int8_t *)${data_out},
                    (int8_t *)${l1_g_ping}, (int8_t *)${l1_g_pong}, (int8_t *)${l1_f_ping}, (int8_t *)${l1_f_pong},
                    (int8_t *)${l1_out}, ${dir}, (float)(-(${g_zero_point})), (float)(${g_scale}),
                    (float)(-(${f_zero_point})), (float)(${f_scale}), (float)(${out_zero_point}), (float)(${out_scale}));
""")

scatterQuantTemplate = _ScatterQuantTemplate("""
// Fused scatter add + quant, int16/int8 in / int8 out (Name: ${nodeName}, Op: ${nodeOp})
GAP9_ScatterQuant_s16_s8((int16_t *)${data_in_net}, (int8_t *)${data_in_agg}, (int32_t *)${data_in_kk}, (int8_t *)${data_out},
                         (int16_t *)${l1_net_ping}, (int16_t *)${l1_net_pong}, (int8_t *)${l1_out_ping}, (int8_t *)${l1_out_pong},
                         (int8_t *)${l1_agg}, ${dir}, (float)(-(${net_zero_point})), (float)(${net_scale}),
                         (float)(-(${agg_zero_point})), (float)(${agg_scale}), (float)(${out_zero_point}), (float)(${out_scale}));
""")

scatterScatterRQLNTemplate = _ScatterScatterRQLNTemplate("""
// Fused scatter add x2 + LayerNorm + quant, int16/int8 in / int8 out (Name: ${nodeName}, Op: ${nodeOp})
GAP9_ScatterScatterRQLN_s16_s8((int16_t *)${data_in_net}, (int8_t *)${data_in_agg1}, (int8_t *)${data_in_agg2},
                               (int32_t *)${data_in_kk}, (float32_t *)${weight}, (float32_t *)${bias}, (int8_t *)${data_out},
                               (int16_t *)${l1_net_ping}, (int16_t *)${l1_net_pong}, (int8_t *)${l1_out_ping}, (int8_t *)${l1_out_pong},
                               (int8_t *)${l1_agg1}, (int8_t *)${l1_agg2}, (float32_t *)${l1_weight}, (float32_t *)${l1_bias},
                               ${dir1}, ${dir2}, (float)(-(${net_zero_point})), (float)(${net_scale}),
                               (float)(-(${agg1_zero_point})), (float)(${agg1_scale}), (float)(-(${agg2_zero_point})), (float)(${agg2_scale}),
                               ${epsilon}, (float)(${out_zero_point}), (float)(${out_scale}));
""")
