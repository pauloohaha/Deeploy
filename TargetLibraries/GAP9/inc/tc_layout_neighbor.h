/*
Author: Pu Deng <pudeng@iis.ethz.ch>
*/


#ifndef __TC_LAYOUT_NEIGHBOR_GAP9__
#define __TC_LAYOUT_NEIGHBOR_GAP9__
#include "CNN_BasicKernels_fp32.h"
#define MAX_PATCH_PER_FRAME  4
#define MAX_EDGE_PER_PATCH 3
#define DIM   ( 384 )


typedef struct {
  float *__restrict__ In;           /**< Pointer to input net tile */
  float *__restrict__ Out;       /**< Pointer to input agg tile */
  int * kk_buff;                /** KK buffer */
  int dir;         /** 0 for previous, 1 for next */
} TC_layout_neighbor_gather_T;



void TC_layout_neighbor_gather(TC_layout_neighbor_gather_T *Arg);
#endif