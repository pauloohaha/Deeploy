/*
 * SPDX-FileCopyrightText: 2025 ETH Zurich and University of Bologna
 *
 * SPDX-License-Identifier: Apache-2.0
 */

/*
 * Add followed by a uniform RequantShift, fused (bit-identical to the Generic int8 + int8 -> int16 Add followed by
 * UniformRequantShift_s16_s8): the sum is kept in a register, never stored, and the work is split over all cores.
 *   out = clamp(((a + b + input_offset) * mul + add + rounding * 2^(log2D-1)) >> log2D + output_offset, min, max)
 */

#include "pmsis.h"

#include "DeeployPULPMath.h"
#include "AddRequantShift.h"

void GAP9_AddRQS_s8_s8_s8(int8_t *data_in_1, int8_t *data_in_2, int8_t *data_out, uint32_t size, int32_t mul,
                          int32_t add, int32_t log2D, int32_t input_offset, int32_t output_offset, int32_t output_min,
                          int32_t output_max, int32_t rounding) {
  int8_t core_id = pi_core_id();
  int8_t log2Core = LOG2(NUM_CORES);
  uint32_t chunk = (size >> log2Core) + ((size & (NUM_CORES - 1)) != 0);
  uint32_t start = MIN(chunk * core_id, size);
  uint32_t stop = MIN(start + chunk, size);
  int32_t round_add = (1 << (log2D - 1)) * rounding;

  for (uint32_t i = start; i < stop; i++) {
    int32_t v = ((int32_t)data_in_1[i] + (int32_t)data_in_2[i] + input_offset) * mul + add;
    v = ((v + round_add) >> log2D) + output_offset;
    data_out[i] = (int8_t)CLAMP(v, output_min, output_max);
  }
}
