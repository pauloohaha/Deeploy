/*
 * SPDX-FileCopyrightText: 2025 ETH Zurich and University of Bologna
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef __DEEPLOY_GAP9_ADD_REQUANT_SHIFT_H__
#define __DEEPLOY_GAP9_ADD_REQUANT_SHIFT_H__

#include <stdint.h>

/* int8 + int8 -> uniform RequantShift -> int8, called by all cores */
void GAP9_AddRQS_s8_s8_s8(int8_t *data_in_1, int8_t *data_in_2, int8_t *data_out, uint32_t size, int32_t mul,
                          int32_t add, int32_t log2D, int32_t input_offset, int32_t output_offset, int32_t output_min,
                          int32_t output_max, int32_t rounding);

#endif
