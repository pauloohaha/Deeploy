/*
Author: Pu Deng <pudeng@iis.ethz.ch>
*/


#ifndef __TC_LAYOUT_NEIGHBOR_GAP9__
#define __TC_LAYOUT_NEIGHBOR_GAP9__
#include "CNN_BasicKernels_fp32.h"
#define MAX_PATCH_PER_FRAME  24
#define MAX_EDGE_PER_PATCH  19


typedef struct {
  int8_t *__restrict__ In;           /**< Pointer to int8 input */
  int8_t *__restrict__ Out;          /**< Pointer to int8 output */
  int * kk_buff;                     /** KK buffer */
  int dir;                           /** 0 for previous, 1 for next */
  int dim;
} TC_layout_neighbor_gather_int8_T;

void TC_layout_neighbor_gather_int8(TC_layout_neighbor_gather_int8_T *Arg);

/* Gather with the preceding uniform RequantShift fused into the copy (bit-identical to RequantShift -> gather):
 *   out = clamp(((in + input_offset) * mul + add + rounding * 2^(log2D-1)) >> log2D + output_offset, min, max)
 * Rows the plain gather zero-fills stay 0 (the requantization of 0 with add = 0 is 0 as well). */
typedef struct {
  int8_t *__restrict__ In;
  int8_t *__restrict__ Out;
  int * kk_buff;
  int dir;
  int dim;
  int32_t mul;
  int32_t add;
  int32_t log2D;
  int32_t input_offset;
  int32_t output_offset;
  int32_t output_min;
  int32_t output_max;
  int32_t rounding;
} TC_layout_neighbor_gather_rqs_int8_T;
void TC_layout_neighbor_gather_rqs_int8(TC_layout_neighbor_gather_rqs_int8_T *Arg);

#endif