# SPDX-FileCopyrightText: 2025 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0

from typing import Dict, List, Tuple, Union

from Deeploy.DeeployTypes import NetworkContext, NodeTemplate, OperatorRepresentation


class FP16LayernormTemplate(NodeTemplate):

    def __init__(self, templateStr):
        super().__init__(templateStr)

    @staticmethod
    def computeTransientBuffersSize(ctxt: NetworkContext,
                                    operatorRepresentation: OperatorRepresentation) -> List[Tuple[str, Union[int]]]:
        reduct_name = "_" + operatorRepresentation['nodeName'] + "_Reduct"
        H = int(operatorRepresentation['size']) // int(operatorRepresentation['lastDimLength'])
        meanbuf_name = "_" + operatorRepresentation['nodeName'] + "_MeanBuf"
        return [
            (reduct_name, int(8 * 4)),   # float32 * 8 cores = 32 bytes
            (meanbuf_name, int(H * 2)),  # float16 * H
        ]

    def hoistTransientBuffers(self, ctxt: NetworkContext,
                              operatorRepresentation: OperatorRepresentation) -> Tuple[NetworkContext, Dict, List[str]]:
        # Reduct: float32 * 8 cores = 32 bytes
        reduct_name = "_"+operatorRepresentation['nodeName'] + "_Reduct"
        ctxt.hoistTransientBuffer(reduct_name, 8 * 4)
        ctxt.lookup(reduct_name)._targetmemorylevel['LayerNormalization'] = "L1"
        operatorRepresentation['Reduct'] = reduct_name

        # MeanBuf: float16 * H (shared for MeanIn/MeanOut)
        H = operatorRepresentation['size'] // operatorRepresentation['lastDimLength']
        meanbuf_name = "_"+operatorRepresentation['nodeName'] + "_MeanBuf"
        ctxt.hoistTransientBuffer(meanbuf_name, H * 2)
        ctxt.lookup(meanbuf_name)._targetmemorylevel['LayerNormalization'] = "L1"
        operatorRepresentation['MeanBuf'] = meanbuf_name

        return ctxt, operatorRepresentation, [reduct_name, meanbuf_name]

    def alignToContext(self, ctxt: NetworkContext,
                       operatorRepresentation: OperatorRepresentation) -> Tuple[NetworkContext, Dict, List[str]]:
        operatorRepresentation['H'] = operatorRepresentation['size'] // operatorRepresentation['lastDimLength']
        operatorRepresentation['W'] = operatorRepresentation['lastDimLength']
        return ctxt, operatorRepresentation, []


referenceTemplate = FP16LayernormTemplate("""
// FP16 LayerNorm (Name: ${nodeName}, Op: ${nodeOp})
{
    KerLayerNorm1D_fp16_T _ln_${nodeName}_arg = {
        .In      = (F16 *)${data_in},
        .H       = ${H},
        .W       = ${W},
        .Reduct  = (float *)${Reduct},
        .Out     = (F16 *)${data_out},
        .MeanIn  = (F16 *)${MeanBuf},
        .MeanOut = (F16 *)${MeanBuf},
        .Weight  = (F16 *)${weight},
        .Bias    = (F16 *)${bias},
        .Epsilon = ${epsilon}f,
    };
    DeeployLayerNorm1D_fp16(&_ln_${nodeName}_arg);
}
""")
