# SPDX-FileCopyrightText: 2023 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0

from typing import Dict, List, Tuple, Union

import numpy as np
from ortools.constraint_solver.pywrapcp import IntVar

from Deeploy.AbstractDataTypes import PointerClass
from Deeploy.CommonExtensions.DataTypes import uint32_t, uint16_t
from Deeploy.DeeployTypes import NetworkContext, OperatorRepresentation
from Deeploy.TilingExtension.MemoryConstraints import NodeMemoryConstraint
from Deeploy.TilingExtension.TileConstraint import TileConstraint
from Deeploy.TilingExtension.TilerModel import TilerModel
from Deeploy.TilingExtension.TilingCodegen import AbsoluteHyperRectangle, TilingSchedule, VariableReplacementScheme, HyperRectangle


class TCneighborGatherConstrain(TileConstraint):

    @staticmethod
    def addGeometricalConstraint(tilerModel: TilerModel, parseDict: Dict, ctxt: NetworkContext) -> TilerModel:
        inputNetBufferName  = parseDict['data_in_net']
        inputKKBufferName   = parseDict['data_in_kk']
        outputBufferName    = parseDict['data_out']

        shapeLen = len(ctxt.lookup(inputNetBufferName).shape)

        # Add I/O dimensions to the model as variables
        for bufferName in [inputNetBufferName, inputKKBufferName, outputBufferName]:
            tilerModel.addTensorDimToModel(ctxt, bufferName)

        # output shape equals to input net shape
        for idx in range(shapeLen):
            outputDim = tilerModel.getTensorDimVar(tensorName = outputBufferName, dimIdx = idx)
            inputDim = tilerModel.getTensorDimVar(tensorName = inputNetBufferName, dimIdx = idx)
            tilerModel.addConstraint(outputDim == inputDim)


        return tilerModel

    @staticmethod
    def addPolicyConstraint(tilerModel: TilerModel, parseDict: Dict, ctxt: NetworkContext) -> TilerModel:
        inputNetBufferName  = parseDict['data_in_net']
        inputKKBufferName   = parseDict['data_in_kk']
        inputNetBuffer  = ctxt.lookup(inputNetBufferName)
        inputKKBuffer   = ctxt.lookup(inputKKBufferName)

        # Get the full size of kk's dimension 0
        kkDim0FullSize = inputKKBuffer.shape[0]

        # Get the tiling variable for kk's dimension 0
        kkDim0Var = tilerModel.getTensorDimVar(tensorName=inputKKBufferName, dimIdx=0)

        # tilerModel.addTensorNumOfEltToModel(ctxt, inputBufferName)
        # numVars = tilerModel.getTensorNumberOfEltVar(inputBufferName)

        # tilerModel.addMinTileSizeConstraint(parseDict, 'size', numVars, 8*lastDimLength)
        # don't tile kk
        tilerModel.addConstraint(kkDim0FullSize == kkDim0Var)

        # don't tile edge dir
        netDim1FullSize = inputNetBuffer.shape[1]
        netDim1Var = tilerModel.getTensorDimVar(tensorName=inputNetBufferName, dimIdx=1)
        tilerModel.addConstraint(netDim1FullSize == netDim1Var)

        return tilerModel

    # @staticmethod
    # def constructSymbolicNodeRep(tilerModel: TilerModel, parseDict: Dict,
    #                              ctxt: NetworkContext) -> Dict[str, Union[int, IntVar]]:

    #     inputBufferName = parseDict['data_in']
    #     inputBuffer = ctxt.lookup(inputBufferName)

    #     lastDimIdx = len(inputBuffer.shape) - 1

    #     symbolicParseDict = parseDict.copy()
    #     symbolicParseDict['lastDimLength'] = tilerModel.getTensorDimVar(inputBuffer.name, lastDimIdx)

    #     return symbolicParseDict

    @classmethod
    def serializeTilingSolution(
            cls, tilingSolution: NodeMemoryConstraint, absoluteOutputCubes: List[AbsoluteHyperRectangle],
            targetMemLevel: str, ctxt: NetworkContext,
            operatorRepresentation: OperatorRepresentation) -> Tuple[VariableReplacementScheme, TilingSchedule]:
        outputCubes = [cube.rectangle for cube in absoluteOutputCubes]

        addrNames = ["data_in_net", "data_in_kk", "data_out"]
        inputBaseOffsets, outputBaseOffsets = cls.extractBaseAddr(tilingSolution, targetMemLevel,
                                                                  operatorRepresentation, addrNames)

        replacements = {}

        replacementTypes = {}

        # for cube in outputCubes:
        #     newSize = np.prod(cube.dims)
        #     replacements["size"].append(newSize)

        inputLoadSchedule = []
        outputLoadSchedule = []

        for cube in outputCubes:
            inputLoadSchedule.append({"data_in_net": cube, "data_in_kk": HyperRectangle(tuple([0]), tuple([2]))})

        for out in outputCubes:
            outputLoadSchedule.append({"data_out": out})

        tilingSchedule = TilingSchedule(inputBaseOffsets, outputBaseOffsets, inputLoadSchedule, outputLoadSchedule)
        variableReplacementSchedule = VariableReplacementScheme(replacements, replacementTypes)

        return variableReplacementSchedule, tilingSchedule

