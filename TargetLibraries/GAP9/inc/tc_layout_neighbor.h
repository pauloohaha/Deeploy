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

#endif