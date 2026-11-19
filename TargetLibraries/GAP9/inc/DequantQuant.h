/*
 * SPDX-FileCopyrightText: 2025 ETH Zurich and University of Bologna
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef __DEEPLOY_GAP9_DEQUANT_QUANT_H__
#define __DEEPLOY_GAP9_DEQUANT_QUANT_H__

#include <stdint.h>

#include "types.h"

/* Dequant -> Quant in fp32, bit-identical to the SDK kernel pair; called by all cores */
#define DQQ_DECLARE(NAME, IN_T, OUT_T)                                                                             \
  void NAME(IN_T *data_in, OUT_T *data_out, uint32_t size, float32_t in_zero_diff, float32_t in_scale,            \
            float32_t out_zero_point, float32_t out_scale);

DQQ_DECLARE(GAP9_DequantQuant_s8_s8, int8_t, int8_t)
DQQ_DECLARE(GAP9_DequantQuant_s8_u8, int8_t, uint8_t)
DQQ_DECLARE(GAP9_DequantQuant_u8_s8, uint8_t, int8_t)
DQQ_DECLARE(GAP9_DequantQuant_u8_u8, uint8_t, uint8_t)
DQQ_DECLARE(GAP9_DequantQuant_s16_s8, int16_t, int8_t)
DQQ_DECLARE(GAP9_DequantQuant_s16_u8, int16_t, uint8_t)

#undef DQQ_DECLARE

#endif
