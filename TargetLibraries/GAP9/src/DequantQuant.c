/*
 * SPDX-FileCopyrightText: 2025 ETH Zurich and University of Bologna
 *
 * SPDX-License-Identifier: Apache-2.0
 */

/*
 * Dequant -> Quant fused, in fp32 with exactly the operations of the GAP9 SDK kernels (bit-identical to the
 * unfused pair, unlike an integer RequantShift approximation of it):
 *   dequant (CNN_FpsFloat32):   x = ((float)q + in_zero_diff) * in_scale
 *   quant (CNN_Float32(U)Fps):  q = trunc(clip(fma(x, out_scale, out_zero_point + 0.5)))
 */

#pragma GCC optimize("fp-contract=off")

#include "pmsis.h"

#include "DeeployPULPMath.h"
#include "DequantQuant.h"

#include <math.h>

#define DQQ_DEFINE(NAME, IN_T, OUT_T, OUT_MIN, OUT_MAX, CVT_T)                                                      \
  void NAME(IN_T *data_in, OUT_T *data_out, uint32_t size, float32_t in_zero_diff, float32_t in_scale,            \
            float32_t out_zero_point, float32_t out_scale) {                                                       \
    int8_t core_id = pi_core_id();                                                                                 \
    int8_t log2Core = LOG2(NUM_CORES);                                                                             \
    uint32_t chunk = (size >> log2Core) + ((size & (NUM_CORES - 1)) != 0);                                         \
    uint32_t start = MIN(chunk * core_id, size);                                                                   \
    uint32_t stop = MIN(start + chunk, size);                                                                      \
    const float32_t out_zero_diff = out_zero_point + 0.5f;                                                         \
    for (uint32_t i = start; i < stop; i++) {                                                                      \
      float32_t x = ((float32_t)data_in[i] + in_zero_diff) * in_scale;                                             \
      float32_t v = fmaf(x, out_scale, out_zero_diff);                                                             \
      v = fminf(v, (float32_t)(OUT_MAX));                                                                          \
      v = fmaxf(v, (float32_t)(OUT_MIN));                                                                          \
      data_out[i] = (OUT_T)(CVT_T)v;                                                                               \
    }                                                                                                              \
  }

DQQ_DEFINE(GAP9_DequantQuant_s8_s8, int8_t, int8_t, -128, 127, int32_t)
DQQ_DEFINE(GAP9_DequantQuant_s8_u8, int8_t, uint8_t, 0, 255, uint32_t)
DQQ_DEFINE(GAP9_DequantQuant_u8_s8, uint8_t, int8_t, -128, 127, int32_t)
DQQ_DEFINE(GAP9_DequantQuant_u8_u8, uint8_t, uint8_t, 0, 255, uint32_t)
DQQ_DEFINE(GAP9_DequantQuant_s16_s8, int16_t, int8_t, -128, 127, int32_t)
DQQ_DEFINE(GAP9_DequantQuant_s16_u8, int16_t, uint8_t, 0, 255, uint32_t)
