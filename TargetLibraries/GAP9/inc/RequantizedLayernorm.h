/*
 * SPDX-FileCopyrightText: 2025 ETH Zurich and University of Bologna
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef __DEEPLOY_GAP9_REQUANTIZED_LAYERNORM_H__
#define __DEEPLOY_GAP9_REQUANTIZED_LAYERNORM_H__

#include <stdint.h>

#include "types.h"

/* Dequant -> LayerNorm (fp32) -> Quant in one kernel, bit-identical to the unfused chain.
 * Called by all cores (rows are split between cores).
 *   x   = ((float)q_in + in_zero_diff) * in_scale          in_zero_diff = -(input zero point)
 *   y   = LayerNorm(x) * scale + bias                      over the last dimension, epsilon as in the graph
 *   out = trunc(clip(y * out_scale + out_zero_point + 0.5)) out_scale = Quant multiplier (1 / step)        */
#define RQLN_DECLARE(NAME, IN_T, OUT_T)                                                                            \
  void NAME(IN_T *data_in, OUT_T *data_out, float32_t *scale, float32_t *bias, uint32_t size,                      \
            uint32_t lastDimLength, float32_t epsilon, float32_t in_zero_diff, float32_t in_scale,                 \
            float32_t out_zero_point, float32_t out_scale);

RQLN_DECLARE(GAP9_RQLayernorm_s8_s8, int8_t, int8_t)
RQLN_DECLARE(GAP9_RQLayernorm_s8_u8, int8_t, uint8_t)
RQLN_DECLARE(GAP9_RQLayernorm_u8_s8, uint8_t, int8_t)
RQLN_DECLARE(GAP9_RQLayernorm_u8_u8, uint8_t, uint8_t)
RQLN_DECLARE(GAP9_RQLayernorm_s16_s8, int16_t, int8_t)
RQLN_DECLARE(GAP9_RQLayernorm_s16_u8, int16_t, uint8_t)

#undef RQLN_DECLARE

#endif
