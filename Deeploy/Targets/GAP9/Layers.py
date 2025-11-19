# SPDX-FileCopyrightText: 2025 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0

from typing import List, Tuple

from Deeploy.DeeployTypes import NodeMapper, Shape, ONNXLayer


    
class CustomSoftmaxAgg(ONNXLayer):

    def __init__(self, maps: List[NodeMapper]):
        super().__init__(maps)
