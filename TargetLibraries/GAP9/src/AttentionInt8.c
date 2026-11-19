/*
 * SPDX-FileCopyrightText: 2025 ETH Zurich and University of Bologna
 *
 * SPDX-License-Identifier: Apache-2.0
 */

/*
 * Fused integer-IO softmax aggregation and scatter kernels, see AttentionInt8.h.
 *
 * Bit-identical to the unfused chain. Contraction of a*b+c is disabled in this file and every operation is
 * written the way the unfused kernels compile:
 *   SDK Dequant (CNN_FpsFloat32, CNN_FpFloat32):  x = ((float)q + zero_diff) * scale
 *   ColSoftMax (KerColSoftMax_fp32):  M = max over rows (start -MAX_FLT32); e = fastexp_clamped(x - M), which
 *       compiles to clip(-88, 88) then (int)fmadd(x, F32_FACA, F32_FACB); Sum += e (from 0); w = e * (1 / Sum)
 *   ElementMul:  p = w * f
 *   ColSum (KerColSum_fp32):  acc = p[row 0]; acc += p[row] for the following rows
 *   ColScatter (KerCatter_fp32):  y = x + agg
 *   LayerNorm (PULP_Layernorm_fp32_fp32): as GAP9_RQLayernorm (RequantizedLayernorm.c)
 *   SDK Quant (CNN_Float32Fps):  q = trunc(clip(fma(y, scale, zero_point + 0.5), -128, 127))
 */

#pragma GCC optimize("fp-contract=off")

#include "pmsis.h"

#include "DeeployPULPMath.h"
#include "column_softmax_gap9.h"
#include "AttentionInt8.h"

#include <math.h>

#define ATT_MAX_ROWS (MAX_PATCH_PER_FRAME > MAX_EDGE_PER_PATCH ? MAX_PATCH_PER_FRAME : MAX_EDGE_PER_PATCH)

static inline float32_t __attribute__((always_inline)) att_exp(float32_t x) {
  union {
    float32_t f;
    int32_t i;
  } r;
  x = clip_f32(x, -88, 88);
  r.i = (int32_t)fmaf(x, F32_FACA, F32_FACB);
  return r.f;
}

static inline int8_t __attribute__((always_inline)) att_quant(float32_t y, float32_t scale, float32_t zero_diff) {
  float32_t v = fmaf(y, scale, zero_diff);
  v = fminf(v, 127.0f);
  v = fmaxf(v, -128.0f);
  return (int8_t)(int32_t)v;
}

/* ------------------------------------------------------------------------------------------------------ */
/* ColAttnPool                                                                                              */
/* ------------------------------------------------------------------------------------------------------ */

typedef struct {
  int8_t *g;   /* [rows][DIM] in L1 */
  int8_t *f;   /* [rows][DIM] in L1 */
  int8_t *out; /* [DIM] in L1 */
  int32_t rows;
  float32_t g_zero_diff, g_scale, f_zero_diff, f_scale, out_zero_diff, out_scale;
} att_pool_args_t;

static void att_pool_worker(void *arg) {
  att_pool_args_t *a = (att_pool_args_t *)arg;
  int32_t core = pi_core_id();
  int32_t ncores = pi_cl_cluster_nb_cores();
  float32_t e[ATT_MAX_ROWS];

  for (int32_t col = core; col < DIM; col += ncores) {
    float32_t M = -MAX_FLT32;
    for (int32_t r = 0; r < a->rows; r++) {
      float32_t x = ((float32_t)a->g[r * DIM + col] + a->g_zero_diff) * a->g_scale;
      M = fmaxf(x, M);
    }
    float32_t Sum = 0.0f;
    for (int32_t r = 0; r < a->rows; r++) {
      float32_t x = ((float32_t)a->g[r * DIM + col] + a->g_zero_diff) * a->g_scale;
      e[r] = att_exp(x - M);
      Sum += e[r];
    }
    float32_t InvSum = 1.0f / Sum;
    float32_t acc = 0.0f;
    for (int32_t r = 0; r < a->rows; r++) {
      float32_t w = e[r] * InvSum;
      float32_t fv = ((float32_t)a->f[r * DIM + col] + a->f_zero_diff) * a->f_scale;
      float32_t p = w * fv;
      acc = (r == 0) ? p : acc + p;
    }
    a->out[col] = att_quant(acc, a->out_scale, a->out_zero_diff);
  }
}

/* Rows of one group: 2D transfer of `rows` rows of DIM bytes, `stride` bytes apart */
static inline void att_group_dma(int8_t *src, int8_t *dst, int32_t rows, int32_t stride, pi_cl_dma_cmd_t *cmd) {
  pi_cl_dma_cmd_2d((uint32_t)src, (uint32_t)dst, rows * DIM, stride, DIM, PI_CL_DMA_DIR_EXT2LOC, cmd);
}

void GAP9_ColAttnPool_s8(int8_t *g, int8_t *f, int32_t *kk, int8_t *out, int8_t *L1_g_ping, int8_t *L1_g_pong,
                         int8_t *L1_f_ping, int8_t *L1_f_pong, int8_t *L1_out, int32_t dir, float32_t g_zero_diff,
                         float32_t g_scale, float32_t f_zero_diff, float32_t f_scale, float32_t out_zero_point,
                         float32_t out_scale) {
  int32_t num_patches = kk[0];
  int32_t num_dst_frames = kk[1];
  if (num_dst_frames == 0 || num_patches == 0) {
    return;
  }

  /* dir 0: one group per patch, its rows are the destination frames (stride one frame)
   * dir 1: one group per destination frame, its rows are the patches (contiguous) */
  int32_t groups = (dir == 0) ? num_patches : num_dst_frames;
  int32_t rows = (dir == 0) ? num_dst_frames : num_patches;
  int32_t row_stride = (dir == 0) ? MAX_PATCH_PER_FRAME * DIM : DIM;
  int32_t group_offset = (dir == 0) ? DIM : MAX_PATCH_PER_FRAME * DIM;

  int8_t *L1_g[2] = {L1_g_ping, L1_g_pong};
  int8_t *L1_f[2] = {L1_f_ping, L1_f_pong};
  pi_cl_dma_cmd_t g_cmd[2], f_cmd[2];

  att_group_dma(g, L1_g[0], rows, row_stride, &g_cmd[0]);
  att_group_dma(f, L1_f[0], rows, row_stride, &f_cmd[0]);

  att_pool_args_t args = {.rows = rows,
                          .g_zero_diff = g_zero_diff,
                          .g_scale = g_scale,
                          .f_zero_diff = f_zero_diff,
                          .f_scale = f_scale,
                          .out_zero_diff = out_zero_point + 0.5f,
                          .out_scale = out_scale};

  for (int32_t grp = 0; grp < groups; grp++) {
    int32_t cur = grp & 1;
    if (grp + 1 < groups) {
      att_group_dma(g + (grp + 1) * group_offset, L1_g[cur ^ 1], rows, row_stride, &g_cmd[cur ^ 1]);
      att_group_dma(f + (grp + 1) * group_offset, L1_f[cur ^ 1], rows, row_stride, &f_cmd[cur ^ 1]);
    }
    pi_cl_dma_cmd_wait(&g_cmd[cur]);
    pi_cl_dma_cmd_wait(&f_cmd[cur]);

    args.g = L1_g[cur];
    args.f = L1_f[cur];
    args.out = L1_out + grp * DIM;
    pi_cl_team_fork(pi_cl_cluster_nb_cores(), att_pool_worker, (void *)&args);
  }

  pi_cl_dma_cmd_t out_cmd;
  pi_cl_dma_cmd((uint32_t)out, (uint32_t)L1_out, groups * DIM, PI_CL_DMA_DIR_LOC2EXT, &out_cmd);
  pi_cl_dma_cmd_wait(&out_cmd);
}

/* ------------------------------------------------------------------------------------------------------ */
/* Scatter kernels: processed one destination frame (MAX_PATCH_PER_FRAME contiguous edges) at a time       */
/* ------------------------------------------------------------------------------------------------------ */

typedef struct {
  int16_t *net; /* [MAX_PATCH_PER_FRAME][DIM] in L1 */
  int8_t *out;  /* [MAX_PATCH_PER_FRAME][DIM] in L1 */
  int8_t *agg1, *agg2;
  float32_t *weight, *bias;
  int32_t frame;          /* destination frame d of this block */
  int32_t num_patches;    /* edges with p < num_patches ... */
  int32_t num_dst_frames; /* ... and d < num_dst_frames take part in the scatter adds */
  int32_t frame_valid;    /* frame < num_dst_frames */
  int32_t dir1, dir2;
  float32_t net_zero_diff, net_scale, agg1_zero_diff, agg1_scale, agg2_zero_diff, agg2_scale;
  float32_t epsilon, out_zero_diff, out_scale;
} att_scatter_args_t;

static inline int32_t __attribute__((always_inline)) att_group_row(int32_t dir, int32_t frame, int32_t patch) {
  return (dir == 0) ? patch : frame;
}

static void att_scatter_quant_worker(void *arg) {
  att_scatter_args_t *a = (att_scatter_args_t *)arg;
  int32_t core = pi_core_id();
  int32_t ncores = pi_cl_cluster_nb_cores();
  int32_t size = MAX_PATCH_PER_FRAME * DIM;
  int32_t chunk = (size + ncores - 1) / ncores;
  int32_t start = MIN(core * chunk, size);
  int32_t stop = MIN(start + chunk, size);

  for (int32_t i = start; i < stop; i++) {
    int32_t p = i / DIM;
    int32_t c = i - p * DIM;
    float32_t y = ((float32_t)a->net[i] + a->net_zero_diff) * a->net_scale;
    if (a->frame_valid && p < a->num_patches) {
      float32_t s = ((float32_t)a->agg1[att_group_row(a->dir1, a->frame, p) * DIM + c] + a->agg1_zero_diff) *
                    a->agg1_scale;
      y = y + s;
    }
    a->out[i] = att_quant(y, a->out_scale, a->out_zero_diff);
  }
}

static void att_scatter_scatter_rqln_worker(void *arg) {
  att_scatter_args_t *a = (att_scatter_args_t *)arg;
  int32_t core = pi_core_id();
  int32_t ncores = pi_cl_cluster_nb_cores();
  float32_t z[DIM];

  for (int32_t p = core; p < MAX_PATCH_PER_FRAME; p += ncores) {
    int16_t *in = a->net + p * DIM;
    int8_t *out = a->out + p * DIM;
    int32_t valid = a->frame_valid && p < a->num_patches;
    int8_t *agg1 = a->agg1 + att_group_row(a->dir1, a->frame, p) * DIM;
    int8_t *agg2 = a->agg2 + att_group_row(a->dir2, a->frame, p) * DIM;

    float32_t mean = 0.0f;
    for (int32_t j = 0; j < DIM; j++) {
      float32_t x = ((float32_t)in[j] + a->net_zero_diff) * a->net_scale;
      if (valid) {
        x = x + ((float32_t)agg1[j] + a->agg1_zero_diff) * a->agg1_scale;
        x = x + ((float32_t)agg2[j] + a->agg2_zero_diff) * a->agg2_scale;
      }
      z[j] = x;
      mean += x;
    }
    mean = mean / (float32_t)DIM;

    float32_t sum = 0.0f;
    for (int32_t j = 0; j < DIM; j++) {
      float32_t temp = z[j] - mean;
      sum = fmaf(temp, temp, sum);
    }
    sum = sum / (float32_t)DIM;
    sum += a->epsilon;
    float32_t std = sqrtf(sum);

    for (int32_t j = 0; j < DIM; j++) {
      float32_t y = fmaf((z[j] - mean) / std, a->weight[j], a->bias[j]);
      out[j] = att_quant(y, a->out_scale, a->out_zero_diff);
    }
  }
}

/* Runs `worker` over all destination-frame blocks of net, double-buffering the int16 input and int8 output */
static void att_scatter_blocks(int16_t *net, int8_t *out, int16_t *L1_net_ping, int16_t *L1_net_pong,
                               int8_t *L1_out_ping, int8_t *L1_out_pong, int32_t num_blocks,
                               att_scatter_args_t *args, void (*worker)(void *)) {
  int16_t *L1_net[2] = {L1_net_ping, L1_net_pong};
  int8_t *L1_out[2] = {L1_out_ping, L1_out_pong};
  pi_cl_dma_cmd_t in_cmd[2], out_cmd[2];
  const int32_t block = MAX_PATCH_PER_FRAME * DIM;

  pi_cl_dma_cmd((uint32_t)net, (uint32_t)L1_net[0], block * sizeof(int16_t), PI_CL_DMA_DIR_EXT2LOC, &in_cmd[0]);

  for (int32_t d = 0; d < num_blocks; d++) {
    int32_t cur = d & 1;
    if (d + 1 < num_blocks) {
      pi_cl_dma_cmd((uint32_t)(net + (d + 1) * block), (uint32_t)L1_net[cur ^ 1], block * sizeof(int16_t),
                    PI_CL_DMA_DIR_EXT2LOC, &in_cmd[cur ^ 1]);
    }
    pi_cl_dma_cmd_wait(&in_cmd[cur]);
    if (d >= 2) {
      pi_cl_dma_cmd_wait(&out_cmd[cur]); /* the output buffer of block d-2 is free again */
    }

    args->net = L1_net[cur];
    args->out = L1_out[cur];
    args->frame = d;
    args->frame_valid = d < args->num_dst_frames;
    pi_cl_team_fork(pi_cl_cluster_nb_cores(), worker, (void *)args);

    pi_cl_dma_cmd((uint32_t)(out + d * block), (uint32_t)L1_out[cur], block, PI_CL_DMA_DIR_LOC2EXT, &out_cmd[cur]);
  }
  pi_cl_dma_cmd_wait(&out_cmd[(num_blocks - 1) & 1]);
  if (num_blocks >= 2) {
    pi_cl_dma_cmd_wait(&out_cmd[num_blocks & 1]);
  }
}

#define ATT_NUM_BLOCKS MAX_EDGE_PER_PATCH /* destination frames in the [MAX_EDGE_PER_PATCH * MAX_PATCH_PER_FRAME, DIM] layout */

void GAP9_ScatterQuant_s16_s8(int16_t *net, int8_t *agg, int32_t *kk, int8_t *out, int16_t *L1_net_ping,
                              int16_t *L1_net_pong, int8_t *L1_out_ping, int8_t *L1_out_pong, int8_t *L1_agg,
                              int32_t dir, float32_t net_zero_diff, float32_t net_scale, float32_t agg_zero_diff,
                              float32_t agg_scale, float32_t out_zero_point, float32_t out_scale) {
  int32_t agg_rows = (dir == 0) ? MAX_PATCH_PER_FRAME : MAX_EDGE_PER_PATCH;
  pi_cl_dma_cmd_t agg_cmd;
  pi_cl_dma_cmd((uint32_t)agg, (uint32_t)L1_agg, agg_rows * DIM, PI_CL_DMA_DIR_EXT2LOC, &agg_cmd);
  pi_cl_dma_cmd_wait(&agg_cmd);

  att_scatter_args_t args = {.agg1 = L1_agg,
                             .agg2 = L1_agg,
                             .num_patches = kk[0],
                             .num_dst_frames = kk[1],
                             .dir1 = dir,
                             .dir2 = dir,
                             .net_zero_diff = net_zero_diff,
                             .net_scale = net_scale,
                             .agg1_zero_diff = agg_zero_diff,
                             .agg1_scale = agg_scale,
                             .out_zero_diff = out_zero_point + 0.5f,
                             .out_scale = out_scale};
  att_scatter_blocks(net, out, L1_net_ping, L1_net_pong, L1_out_ping, L1_out_pong, ATT_NUM_BLOCKS, &args,
                     att_scatter_quant_worker);
}

void GAP9_ScatterScatterRQLN_s16_s8(int16_t *net, int8_t *agg1, int8_t *agg2, int32_t *kk, float32_t *weight,
                                    float32_t *bias, int8_t *out, int16_t *L1_net_ping, int16_t *L1_net_pong,
                                    int8_t *L1_out_ping, int8_t *L1_out_pong, int8_t *L1_agg1, int8_t *L1_agg2,
                                    float32_t *L1_weight, float32_t *L1_bias, int32_t dir1, int32_t dir2,
                                    float32_t net_zero_diff, float32_t net_scale, float32_t agg1_zero_diff,
                                    float32_t agg1_scale, float32_t agg2_zero_diff, float32_t agg2_scale,
                                    float32_t epsilon, float32_t out_zero_point, float32_t out_scale) {
  int32_t agg1_rows = (dir1 == 0) ? MAX_PATCH_PER_FRAME : MAX_EDGE_PER_PATCH;
  int32_t agg2_rows = (dir2 == 0) ? MAX_PATCH_PER_FRAME : MAX_EDGE_PER_PATCH;
  pi_cl_dma_cmd_t cmd[4];
  pi_cl_dma_cmd((uint32_t)agg1, (uint32_t)L1_agg1, agg1_rows * DIM, PI_CL_DMA_DIR_EXT2LOC, &cmd[0]);
  pi_cl_dma_cmd((uint32_t)agg2, (uint32_t)L1_agg2, agg2_rows * DIM, PI_CL_DMA_DIR_EXT2LOC, &cmd[1]);
  pi_cl_dma_cmd((uint32_t)weight, (uint32_t)L1_weight, DIM * sizeof(float32_t), PI_CL_DMA_DIR_EXT2LOC, &cmd[2]);
  pi_cl_dma_cmd((uint32_t)bias, (uint32_t)L1_bias, DIM * sizeof(float32_t), PI_CL_DMA_DIR_EXT2LOC, &cmd[3]);
  for (int32_t i = 0; i < 4; i++) {
    pi_cl_dma_cmd_wait(&cmd[i]);
  }

  att_scatter_args_t args = {.agg1 = L1_agg1,
                             .agg2 = L1_agg2,
                             .weight = L1_weight,
                             .bias = L1_bias,
                             .num_patches = kk[0],
                             .num_dst_frames = kk[1],
                             .dir1 = dir1,
                             .dir2 = dir2,
                             .net_zero_diff = net_zero_diff,
                             .net_scale = net_scale,
                             .agg1_zero_diff = agg1_zero_diff,
                             .agg1_scale = agg1_scale,
                             .agg2_zero_diff = agg2_zero_diff,
                             .agg2_scale = agg2_scale,
                             .epsilon = epsilon,
                             .out_zero_diff = out_zero_point + 0.5f,
                             .out_scale = out_scale};
  att_scatter_blocks(net, out, L1_net_ping, L1_net_pong, L1_out_ping, L1_out_pong, ATT_NUM_BLOCKS, &args,
                     att_scatter_scatter_rqln_worker);
}
