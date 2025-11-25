/*
Author: Pu Deng <pudeng@iis.ethz.ch>
*/


#ifndef __COLUMN_SOFTMAX_GAP9__
#define __COLUMN_SOFTMAX_GAP9__

#include "at_api.h"
#include "CNN_BasicKernels_fp32.h"

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

void KerColSoftMax8Bits_SQ8 (KerColSoftMax_SQ8_T *Arg);
void KerColSoftMax_fp32 (void * slave_arg);
void SoftMaxAgg_master_kernel(float *L2_net_buffer, int *L2_KK_buffer, float *L2_output_buffer, float *L1_edge_in_buffer, float *L1_edge_out_buffer, int *collected_edge_buffer);
#endif