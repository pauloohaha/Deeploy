/*
Author: Pu Deng <pudeng@iis.ethz.ch>
*/

#include "tc_layout_neighbor.h"


static int CoreCountDynamic = 1;
static int ActiveCore = gap_ncore();

static inline unsigned int __attribute__((always_inline)) ChunkSize(unsigned int X)

{
	unsigned int NCore;
	unsigned int Log2Core;
	unsigned int Chunk;

	if (CoreCountDynamic) NCore = ActiveCore; else NCore = gap_ncore();
	Log2Core = gap_fl1(NCore);
	Chunk = (X>>Log2Core) + ((X&(NCore-1))!=0);
	return Chunk;
}

void TC_layout_neighbor_gather_int8(TC_layout_neighbor_gather_int8_T *Arg) {
    int8_t * In = Arg->In;
    int8_t * Out = Arg->Out;
    int patches    = Arg->kk_buff[0];
    int dst_frames = Arg->kk_buff[1];
    int dir  = Arg->dir;
    int dim = Arg->dim;

    unsigned int CoreId = gap_coreid();
    unsigned int total_quads = (dim * patches) / 4;
    unsigned int Chunk = ChunkSize(total_quads);
    unsigned int First = Chunk * CoreId;
    unsigned int Last = Min(First + Chunk, total_quads);

    for(int out_dst_frame = 0; out_dst_frame < dst_frames-1; out_dst_frame++){
      for(unsigned int q = First; q < Last; q++){
        unsigned int element_id = q * 4;
        unsigned int in_idx = out_dst_frame*MAX_PATCH_PER_FRAME*dim + element_id;
        if(dir == 1) in_idx += MAX_PATCH_PER_FRAME*dim;
        unsigned int out_idx = out_dst_frame*MAX_PATCH_PER_FRAME*dim + element_id;
        if(dir == 0) out_idx += MAX_PATCH_PER_FRAME*dim;
        *((v4s *)&Out[out_idx]) = *((v4s *)&In[in_idx]);
      }
    }

    v4s zero = (v4s){0,0,0,0};
    for(unsigned int q = First; q < Last; q++){
      unsigned int element_id = q * 4;
      unsigned int out_idx = (dir) ? element_id + (dst_frames-1)*MAX_PATCH_PER_FRAME*dim : element_id;
      *((v4s *)&Out[out_idx]) = zero;
    }

    return;
}


static inline int8_t __attribute__((always_inline))
gather_rqs(int8_t x, const TC_layout_neighbor_gather_rqs_int8_T *a, int32_t round_add) {
  int32_t v = (x + a->input_offset) * a->mul + a->add;
  v = ((v + round_add) >> a->log2D) + a->output_offset;
  return (int8_t)(v < a->output_min ? a->output_min : (v > a->output_max ? a->output_max : v));
}

void TC_layout_neighbor_gather_rqs_int8(TC_layout_neighbor_gather_rqs_int8_T *Arg) {
    int8_t * In = Arg->In;
    int8_t * Out = Arg->Out;
    int patches    = Arg->kk_buff[0];
    int dst_frames = Arg->kk_buff[1];
    int dir  = Arg->dir;
    int dim = Arg->dim;
    int32_t round_add = (1 << (Arg->log2D - 1)) * Arg->rounding;

    unsigned int CoreId = gap_coreid();
    unsigned int total_quads = (dim * patches) / 4;
    unsigned int Chunk = ChunkSize(total_quads);
    unsigned int First = Chunk * CoreId;
    unsigned int Last = Min(First + Chunk, total_quads);

    for(int out_dst_frame = 0; out_dst_frame < dst_frames-1; out_dst_frame++){
      for(unsigned int q = First; q < Last; q++){
        unsigned int element_id = q * 4;
        unsigned int in_idx = out_dst_frame*MAX_PATCH_PER_FRAME*dim + element_id;
        if(dir == 1) in_idx += MAX_PATCH_PER_FRAME*dim;
        unsigned int out_idx = out_dst_frame*MAX_PATCH_PER_FRAME*dim + element_id;
        if(dir == 0) out_idx += MAX_PATCH_PER_FRAME*dim;
        v4s x = *((v4s *)&In[in_idx]);
        *((v4s *)&Out[out_idx]) = (v4s){gather_rqs(x[0], Arg, round_add), gather_rqs(x[1], Arg, round_add),
                                        gather_rqs(x[2], Arg, round_add), gather_rqs(x[3], Arg, round_add)};
      }
    }

    v4s zero = (v4s){0,0,0,0};
    for(unsigned int q = First; q < Last; q++){
      unsigned int element_id = q * 4;
      unsigned int out_idx = (dir) ? element_id + (dst_frames-1)*MAX_PATCH_PER_FRAME*dim : element_id;
      *((v4s *)&Out[out_idx]) = zero;
    }
    return;
}
