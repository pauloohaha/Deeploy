# SPDX-FileCopyrightText: 2025 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0

from typing import Dict, List, Tuple

from Deeploy.DeeployTypes import NetworkContext, OperatorRepresentation
from Deeploy.TilingExtension.MemoryConstraints import NodeMemoryConstraint
from Deeploy.TilingExtension.TileConstraint import TileConstraint
from Deeploy.TilingExtension.TilerModel import TilerModel
from Deeploy.TilingExtension.TilingCodegen import AbsoluteHyperRectangle, HyperRectangle, TilingSchedule, \
    VariableReplacementScheme


class WholeTensorTileConstraint(TileConstraint):
    """Every operand is one untiled tile (the kernel tiles internally, like the CustomCol* kernels)."""

    dataInNames: List[str] = []
    dataOutName: str = 'data_out'

    @classmethod
    def addGeometricalConstraint(cls, tilerModel: TilerModel, parseDict: Dict, ctxt: NetworkContext) -> TilerModel:
        for key in cls.dataInNames + [cls.dataOutName]:
            tilerModel.addTensorDimToModel(ctxt, parseDict[key])
        return tilerModel

    @classmethod
    def addPolicyConstraint(cls, tilerModel: TilerModel, parseDict: Dict, ctxt: NetworkContext) -> TilerModel:
        for key in cls.dataInNames + [cls.dataOutName]:
            name = parseDict[key]
            for idx, full in enumerate(ctxt.lookup(name).shape):
                tilerModel.addConstraint(tilerModel.getTensorDimVar(tensorName = name, dimIdx = idx) == full)
        return tilerModel

    @classmethod
    def serializeTilingSolution(
            cls, tilingSolution: NodeMemoryConstraint, absoluteOutputCubes: List[AbsoluteHyperRectangle],
            targetMemLevel: str, ctxt: NetworkContext,
            operatorRepresentation: OperatorRepresentation) -> Tuple[VariableReplacementScheme, TilingSchedule]:
        outputCubes = [cube.rectangle for cube in absoluteOutputCubes]
        addrNames = cls.dataInNames + [cls.dataOutName]
        inputBaseOffsets, outputBaseOffsets = cls.extractBaseAddr(tilingSolution, targetMemLevel,
                                                                  operatorRepresentation, addrNames)
        fullRects = {}
        for key in cls.dataInNames:
            shape = tuple(ctxt.lookup(operatorRepresentation[key]).shape)
            fullRects[key] = HyperRectangle(tuple([0] * len(shape)), shape)
        inputLoadSchedule = [dict(fullRects) for _ in outputCubes]
        outputLoadSchedule = [{cls.dataOutName: cube} for cube in outputCubes]
        tilingSchedule = TilingSchedule(inputBaseOffsets, outputBaseOffsets, inputLoadSchedule, outputLoadSchedule)
        return VariableReplacementScheme({}, {}), tilingSchedule


class ColAttnPoolTileConstraint(WholeTensorTileConstraint):
    dataInNames = ['data_in_g', 'data_in_f', 'data_in_kk']


class ScatterQuantTileConstraint(WholeTensorTileConstraint):
    dataInNames = ['data_in_net', 'data_in_agg', 'data_in_kk']


class ScatterScatterRQLNTileConstraint(WholeTensorTileConstraint):
    dataInNames = ['data_in_net', 'data_in_agg1', 'data_in_agg2', 'data_in_kk', 'weight', 'bias']
