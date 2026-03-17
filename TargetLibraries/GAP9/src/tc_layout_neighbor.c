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

    unsigned int CoreId = gap_coreid();
    unsigned int total_quads = (DIM * patches) / 4;
    unsigned int Chunk = ChunkSize(total_quads);
    unsigned int First = Chunk * CoreId;
    unsigned int Last = Min(First + Chunk, total_quads);

    for(int out_dst_frame = 0; out_dst_frame < dst_frames-1; out_dst_frame++){
      for(unsigned int q = First; q < Last; q++){
        unsigned int element_id = q * 4;
        unsigned int in_idx = out_dst_frame*MAX_PATCH_PER_FRAME*DIM + element_id;
        if(dir == 1) in_idx += MAX_PATCH_PER_FRAME*DIM;
        unsigned int out_idx = out_dst_frame*MAX_PATCH_PER_FRAME*DIM + element_id;
        if(dir == 0) out_idx += MAX_PATCH_PER_FRAME*DIM;
        *((v4s *)&Out[out_idx]) = *((v4s *)&In[in_idx]);
      }
    }

    v4s zero = (v4s){0,0,0,0};
    for(unsigned int q = First; q < Last; q++){
      unsigned int element_id = q * 4;
      unsigned int out_idx = (dir) ? element_id + (dst_frames-1)*MAX_PATCH_PER_FRAME*DIM : element_id;
      *((v4s *)&Out[out_idx]) = zero;
    }

    return;
}

