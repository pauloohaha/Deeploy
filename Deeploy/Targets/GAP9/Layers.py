# SPDX-FileCopyrightText: 2025 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0

from typing import List, Tuple

from Deeploy.DeeployTypes import NodeMapper, Shape, ONNXLayer


    
class CustomColSoftmax(ONNXLayer):

    def __init__(self, maps: List[NodeMapper]):
        super().__init__(maps)


class CustomColSum(ONNXLayer):

    def __init__(self, maps: List[NodeMapper]):
        super().__init__(maps)

class CustomColScatter(ONNXLayer):

    def __init__(self, maps: List[NodeMapper]):
        super().__init__(maps)

class Sigmoid(ONNXLayer):

    def __init__(self, maps: List[NodeMapper]):
        super().__init__(maps)

class TCneighborGather(ONNXLayer):

    def __init__(self, maps: List[NodeMapper]):
        super().__init__(maps)

class Instancenorm2d(ONNXLayer):

    def __init__(self, maps: List[NodeMapper]):
        super().__init__(maps)