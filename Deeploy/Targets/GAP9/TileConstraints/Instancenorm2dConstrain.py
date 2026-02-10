# SPDX-FileCopyrightText: 2024 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0

from typing import Dict, List, Tuple

import numpy as np

from Deeploy.AbstractDataTypes import PointerClass
from Deeploy.CommonExtensions.DataTypes import uint16_t
from Deeploy.DeeployTypes import NetworkContext, OperatorRepresentation
from Deeploy.TilingExtension.MemoryConstraints import NodeMemoryConstraint
from Deeploy.TilingExtension.TileConstraint import TileConstraint
from Deeploy.TilingExtension.TilerModel import TilerModel
from Deeploy.TilingExtension.TilingCodegen import AbsoluteHyperRectangle, HyperRectangle, TilingSchedule, \
    VariableReplacementScheme


class Instancenorm2dTileConstraint(TileConstraint):

    @staticmethod
    def addGeometricalConstraint(tilerModel: TilerModel, parseDict: Dict, ctxt: NetworkContext) -> TilerModel:
        inputBufferName = parseDict['data_in']
        outputBufferName = parseDict['data_out']
        scaleBufferName = parseDict['weight']
        biasBufferName = parseDict['bias']

        for bufferName in [inputBufferName, outputBufferName, scaleBufferName, biasBufferName]:
            tilerModel.addTensorDimToModel(ctxt, bufferName)

        # HxW shoud be untiled
        inputShape = ctxt.lookup(inputBufferName).shape
        WDimIdx = len(inputShape) - 1
        WDimLen = inputShape[-1]
        HDimIdx = len(inputShape) - 2
        HDimLen = inputShape[-2]

        tilerModel.addConstraint(
            tilerModel.getTensorDimVar(tensorName = inputBufferName, dimIdx = WDimIdx) == WDimLen)
        tilerModel.addConstraint(
            tilerModel.getTensorDimVar(tensorName = inputBufferName, dimIdx = HDimIdx) == HDimLen)

        # scale and weight should have the shape of the channel width of input
        channelDimIdx = len(inputShape) - 3
        tilerModel.addConstraint(
            tilerModel.getTensorDimVar(tensorName = inputBufferName, dimIdx = channelDimIdx) == tilerModel.getTensorDimVar(
                tensorName = scaleBufferName, dimIdx = 0))
        tilerModel.addConstraint(
            tilerModel.getTensorDimVar(tensorName = inputBufferName, dimIdx = channelDimIdx) == tilerModel.getTensorDimVar(
                tensorName = biasBufferName, dimIdx = 0))

        # input and output should have the same shape
        for idx, dim in enumerate(inputShape):
            tilerModel.addConstraint(
                tilerModel.getTensorDimVar(tensorName = inputBufferName, dimIdx = idx) == tilerModel.getTensorDimVar(
                    tensorName = outputBufferName, dimIdx = idx))

        return tilerModel

    @classmethod
    def serializeTilingSolution(
            cls, tilingSolution: NodeMemoryConstraint, absoluteOutputCubes: List[AbsoluteHyperRectangle],
            targetMemLevel: str, ctxt: NetworkContext,
            operatorRepresentation: OperatorRepresentation) -> Tuple[VariableReplacementScheme, TilingSchedule]:

        outputCubes = [cube.rectangle for cube in absoluteOutputCubes]
        addrNames = ['data_in', 'data_out', 'weight', 'bias']
        inputBaseOffsets, outputBaseOffsets = cls.extractBaseAddr(tilingSolution, targetMemLevel,
                                                                  operatorRepresentation, addrNames)

        replacements = {"size": []}

        replacementTypes = {"size": PointerClass(uint16_t)}

        inputLoadSchedule = []
        outputLoadSchedule = []

        for cube in outputCubes:
            newSize = np.prod(cube.dims)
            replacements["size"].append(newSize)
            weightCube = HyperRectangle((0,), (cube.dims[-3],)) # weight and bias has the size of channel
            biasCube = HyperRectangle((0,), (cube.dims[-3],))
            inputLoadSchedule.append({"data_in": cube, "weight": weightCube, "bias": biasCube})
            outputLoadSchedule.append({"data_out": cube})

        tilingSchedule = TilingSchedule(inputBaseOffsets, outputBaseOffsets, inputLoadSchedule, outputLoadSchedule)
        variableReplacementSchedule = VariableReplacementScheme(replacements, replacementTypes)

        return variableReplacementSchedule, tilingSchedule
