/*
 * SPDX-FileCopyrightText: 2025 ETH Zurich and University of Bologna
 *
 * SPDX-License-Identifier: Apache-2.0
 */

/*
 * Requantized LayerNorm: Dequant -> LayerNorm (fp32) -> Quant fused into one kernel.
 *
 * Reads the quantized input, dequantizes each element on load, computes the LayerNorm in fp32 with
 * exactly the operations of PULP_Layernorm_fp32_fp32, and quantizes each result on store with exactly the
 * operations of the GAP9 SDK Quant kernels (CNN_Float32Fps / CNN_Float32UFps). The fp32 intermediates of the
 * unfused graph never exist in memory, and the results are bit-identical to the unfused chain:
 *   dequant (CNN_FpsFloat32):   x = ((float)q + in_zero_diff) * in_scale       (add, then multiply)
 *   layernorm:                  mean, var = sum((x - mean)^2) (fused multiply-add), y = ((x-mean)/std)*w + b (fma)
 *   quant (CNN_Float32(U)Fps):  q = trunc(clip(fma(y, out_scale, out_zero_point + 0.5)))
 * Contraction of the remaining a*b+c expressions is disabled so the compiler cannot change the rounding.
 */

#pragma GCC optimize("fp-contract=off")

#include "pmsis.h"

#include "DeeployPULPMath.h"
#include "RequantizedLayernorm.h"

#include <math.h>

#define RQLN_DEFINE(NAME, IN_T, OUT_T, OUT_MIN, OUT_MAX, CVT_T)                                                     \
  void NAME(IN_T *data_in, OUT_T *data_out, float32_t *scale, float32_t *bias, uint32_t size,                      \
            uint32_t lastDimLength, float32_t epsilon, float32_t in_zero_diff, float32_t in_scale,                 \
            float32_t out_zero_point, float32_t out_scale) {                                                       \
    int8_t core_id = pi_core_id();                                                                                 \
    int8_t log2Core = LOG2(NUM_CORES);                                                                             \
                                                                                                                   \
    int32_t seq_length = size / lastDimLength;                                                                     \
    int32_t chunk = (seq_length >> log2Core) + ((seq_length & (NUM_CORES - 1)) != 0);                             \
    int32_t start_seq = MIN(chunk * core_id, seq_length);                                                          \
    int32_t end_seq = MIN(start_seq + chunk, seq_length);                                                          \
                                                                                                                   \
    const float32_t out_zero_diff = out_zero_point + 0.5f;                                                         \
                                                                                                                   \
    for (int32_t s = start_seq; s < end_seq; s++) {                                                                \
      IN_T *in = data_in + s * lastDimLength;                                                                      \
      OUT_T *out = data_out + s * lastDimLength;                                                                   \
                                                                                                                   \
      float32_t mean = 0.0f;                                                                                       \
      for (int32_t j = 0; j < (int32_t)lastDimLength; j++) {                                                       \
        mean += ((float32_t)in[j] + in_zero_diff) * in_scale;                                                     \
      }                                                                                                            \
      mean = mean / (float32_t)lastDimLength;                                                                      \
                                                                                                                   \
      float32_t sum = 0.0f;                                                                                        \
      for (int32_t j = 0; j < (int32_t)lastDimLength; j++) {                                                       \
        float32_t temp = ((float32_t)in[j] + in_zero_diff) * in_scale - mean;                                     \
        sum = fmaf(temp, temp, sum);                                                                               \
      }                                                                                                            \
      sum = sum / (float32_t)lastDimLength;                                                                        \
      sum += epsilon;                                                                                              \
      float32_t std = sqrtf(sum);                                                                                  \
                                                                                                                   \
      for (int32_t j = 0; j < (int32_t)lastDimLength; j++) {                                                       \
        float32_t x = ((float32_t)in[j] + in_zero_diff) * in_scale;                                               \
        float32_t y = fmaf((x - mean) / std, scale[j], bias[j]);                                                   \
        float32_t v = fmaf(y, out_scale, out_zero_diff);                                                           \
        v = fminf(v, (float32_t)(OUT_MAX));                                                                        \
        v = fmaxf(v, (float32_t)(OUT_MIN));                                                                        \
        out[j] = (OUT_T)(CVT_T)v;                                                                                  \
      }                                                                                                            \
    }                                                                                                              \
  }

RQLN_DEFINE(GAP9_RQLayernorm_s8_s8, int8_t, int8_t, -128, 127, int32_t)
RQLN_DEFINE(GAP9_RQLayernorm_s8_u8, int8_t, uint8_t, 0, 255, uint32_t)
RQLN_DEFINE(GAP9_RQLayernorm_u8_s8, uint8_t, int8_t, -128, 127, int32_t)
RQLN_DEFINE(GAP9_RQLayernorm_u8_u8, uint8_t, uint8_t, 0, 255, uint32_t)
RQLN_DEFINE(GAP9_RQLayernorm_s16_s8, int16_t, int8_t, -128, 127, int32_t)
RQLN_DEFINE(GAP9_RQLayernorm_s16_u8, int16_t, uint8_t, 0, 255, uint32_t)
