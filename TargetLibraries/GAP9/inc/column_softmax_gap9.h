/*
Author: Pu Deng <pudeng@iis.ethz.ch>
*/


#ifndef __COLUMN_SOFTMAX_GAP9__
#define __COLUMN_SOFTMAX_GAP9__

#include "at_api.h"
#include "CNN_BasicKernels_fp32.h"

#define MAX_PATCH_PER_FRAME  4
#define MAX_EDGE_PER_PATCH 3
#define DIM   ( 384 )

typedef struct {
  signed char *__restrict__ In;           /**< Pointer to input tile */
  unsigned short int Feat;                /**< Number of features of the tile */
	unsigned short int N;                   /**< Size of the tile */
	unsigned short int Norm;                /**< Normalization factor */
	void *__restrict__ Out;            /**< Pointer to output tile */

} KerColSoftMax_SQ8_T;

typedef struct {
  float *__restrict__ In;           /**< Pointer to input tile */
  unsigned short int Feat;                /**< Number of features of the tile */
	unsigned short int N;                   /**< Size of the tile */
	float *__restrict__ Out;            /**< Pointer to output tile */

} KerColSoftMax_fp32_T;

typedef struct {
  float *__restrict__ In;           /**< Pointer to input net tile */
  float *__restrict__ In_agg;       /**< Pointer to input agg tile */
  unsigned short int Feat;                /**< Number of features of the tile */
	unsigned short int N;                   /**< Size of the tile */
} KerColScatter_fp32_T;

void ColSoftMax_master_kernel(float *L2_net_buffer, int *L2_KK_buffer, float *L2_output_buffer, 
                              float *L1_edge_in_ping_buffer, float *L1_edge_out_ping_buffer, 
                              float *L1_edge_in_pong_buffer, float *L1_edge_out_pong_buffer,
                              int dir);

void ColSum_master_kernel(float *L2_net_buffer, int *L2_KK_buffer, float *L2_output_buffer, 
                              float *L1_edge_in_ping_buffer, float *L1_edge_out_ping_buffer, 
                              float *L1_edge_in_pong_buffer, float *L1_edge_out_pong_buffer,
                              int dir);

void ColScatter_master_kernel(float *L2_net_buffer, int *L2_KK_buffer, float* L2_agg_buffer, float *L2_output_buffer, 
                              float *L1_edge_ping_buffer, float *L1_agg_ping_buffer,
                              float *L1_edge_pong_buffer, float *L1_agg_pong_buffer,
                              int dir);
#endif