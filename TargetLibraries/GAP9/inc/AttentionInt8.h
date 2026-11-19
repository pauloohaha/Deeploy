/*
 * SPDX-FileCopyrightText: 2025 ETH Zurich and University of Bologna
 *
 * SPDX-License-Identifier: Apache-2.0
 */

/*
 * Fused integer-IO kernels for the TinyDEVO update block's softmax aggregation (patch and frame
 * attention) and its scatter adds. Each one reads only integer tensors, dequantizes on load, computes in
 * fp32 with exactly the operations of the unfused float graph (SDK Dequant, ColSoftMax, ElementMul,
 * ColSum, ColScatter, PULP LayerNorm, SDK Quant), and quantizes on store, so the results are bit-identical
 * to the unfused chain while none of its fp32 tensors exist in memory.
 *
 * Edge layout (as in column_softmax_gap9): edge e = d * MAX_PATCH_PER_FRAME + p, row e holds DIM values.
 * Patch aggregation (dir 0) groups the edges of one patch p, frame aggregation (dir 1) those of one
 * destination frame d. Only edges with p < kk[0] (patches) and d < kk[1] (destination frames) take part.
 *
 * All three are master functions: called by one core, they move tiles between L2 and the given L1 buffers
 * with the cluster DMA and fork the computation to all cores.
 */

#ifndef __DEEPLOY_GAP9_ATTENTION_INT8_H__
#define __DEEPLOY_GAP9_ATTENTION_INT8_H__

#include <stdint.h>

#include "types.h"

/* out[group] = Quant( sum_rows( softmax_rows(Dequant(g)) * Dequant(f) ) ), one int8 row of DIM per group.
 * L1 buffers: g/f ping/pong of MAX(MAX_EDGE_PER_PATCH, MAX_PATCH_PER_FRAME) * DIM bytes each, out of
 * MAX(MAX_EDGE_PER_PATCH, MAX_PATCH_PER_FRAME) * DIM bytes. */
void GAP9_ColAttnPool_s8(int8_t *g, int8_t *f, int32_t *kk, int8_t *out, int8_t *L1_g_ping, int8_t *L1_g_pong,
                         int8_t *L1_f_ping, int8_t *L1_f_pong, int8_t *L1_out, int32_t dir, float32_t g_zero_diff,
                         float32_t g_scale, float32_t f_zero_diff, float32_t f_scale, float32_t out_zero_point,
                         float32_t out_scale);

/* out[e] = Quant( Dequant(net[e]) + Dequant(agg[group(e)]) )  (ColScatter of dir followed by Quant).
 * L1 buffers: net ping/pong of MAX_PATCH_PER_FRAME * DIM int16, out ping/pong of MAX_PATCH_PER_FRAME * DIM
 * bytes, agg of MAX(MAX_EDGE_PER_PATCH, MAX_PATCH_PER_FRAME) * DIM bytes. */
void GAP9_ScatterQuant_s16_s8(int16_t *net, int8_t *agg, int32_t *kk, int8_t *out, int16_t *L1_net_ping,
                              int16_t *L1_net_pong, int8_t *L1_out_ping, int8_t *L1_out_pong, int8_t *L1_agg,
                              int32_t dir, float32_t net_zero_diff, float32_t net_scale, float32_t agg_zero_diff,
                              float32_t agg_scale, float32_t out_zero_point, float32_t out_scale);

/* out[e] = Quant( LayerNorm( (Dequant(net[e]) + Dequant(agg1[group1(e)])) + Dequant(agg2[group2(e)]) ) )
 * (ColScatter dir1, ColScatter dir2, LayerNormalization over DIM, Quant).
 * L1 buffers as GAP9_ScatterQuant_s16_s8 plus a second agg buffer and DIM floats each for weight and bias. */
void GAP9_ScatterScatterRQLN_s16_s8(int16_t *net, int8_t *agg1, int8_t *agg2, int32_t *kk, float32_t *weight,
                                    float32_t *bias, int8_t *out, int16_t *L1_net_ping, int16_t *L1_net_pong,
                                    int8_t *L1_out_ping, int8_t *L1_out_pong, int8_t *L1_agg1, int8_t *L1_agg2,
                                    float32_t *L1_weight, float32_t *L1_bias, int32_t dir1, int32_t dir2,
                                    float32_t net_zero_diff, float32_t net_scale, float32_t agg1_zero_diff,
                                    float32_t agg1_scale, float32_t agg2_zero_diff, float32_t agg2_scale,
                                    float32_t epsilon, float32_t out_zero_point, float32_t out_scale);

#endif
