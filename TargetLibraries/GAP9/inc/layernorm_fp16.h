/*
 * SPDX-FileCopyrightText: 2025 ETH Zurich and University of Bologna
 * SPDX-License-Identifier: Apache-2.0
 *
 * FP16 LayerNorm kernel for GAP9 with epsilon, weight (gamma), and bias (beta).
 */

#ifndef __LAYERNORM_FP16_GAP9__
#define __LAYERNORM_FP16_GAP9__

#include "pmsis.h"
#include "at_api.h"
#include "CNN_FloatType.h"

typedef struct {
    F16 * __restrict__ In;       /**< Input data [H x W] */
    unsigned short int H;        /**< Number of rows (sequence length) */
    unsigned short int W;        /**< Feature dimension per row */
    float * __restrict__ Reduct; /**< Per-core reduction buffer [NUM_CORES] */
    F16 * __restrict__ Out;      /**< Output data [H x W] */
    F16 * __restrict__ MeanIn;   /**< Intermediate mean/rstd buffer [H] */
    F16 * __restrict__ MeanOut;  /**< Intermediate mean/rstd buffer [H] */
    F16 * __restrict__ Weight;   /**< Per-feature scale (gamma) [W] */
    F16 * __restrict__ Bias;     /**< Per-feature offset (beta) [W] */
    float Epsilon;               /**< Numerical stability constant */
} KerLayerNorm1D_fp16_T;

void DeeployLayerNorm1D_fp16(KerLayerNorm1D_fp16_T *Arg);

#endif
