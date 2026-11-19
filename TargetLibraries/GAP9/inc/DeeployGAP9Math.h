/*
 * SPDX-FileCopyrightText: 2021 ETH Zurich and University of Bologna
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef __DEEPLOY_MATH_HEADER_
#define __DEEPLOY_MATH_HEADER_

#include <ctype.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#define BEGIN_SINGLE_CORE if (pi_core_id() == 8 || pi_core_id() == 0) {
#define END_SINGLE_CORE }
#define SINGLE_CORE if (pi_core_id() == 8 || pi_core_id() == 0)

#include "DeeployBasicMath.h"

#include "dory_dma.h"
#include "dory_mem.h"
#include "../../PULPOpen/inc/kernel/Layernorm.h"
#include "../../PULPOpen/inc/kernel/UniformRequantShift.h"
#include "column_softmax_gap9.h"
#include "RequantizedLayernorm.h"
#include "AddRequantShift.h"
#include "DequantQuant.h"
#include "AttentionInt8.h"
#include "tc_layout_neighbor.h"

#include "pmsis.h"

#endif // __DEEPLOY_MATH_HEADER_
