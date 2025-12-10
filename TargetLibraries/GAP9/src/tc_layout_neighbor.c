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

void TC_layout_neighbor_gather(TC_layout_neighbor_gather_T *Arg) {
    float * In = Arg->In;
    float * Out = Arg->Out;
    int patches    = Arg->kk_buff[0];
    int dst_frames = Arg->kk_buff[1];
    int dir  = Arg->dir;

    unsigned int CoreId = gap_coreid();
    unsigned int Chunk = ChunkSize(DIM*patches);
    unsigned int First = Chunk*CoreId;
    unsigned int Last = Min(First+Chunk, DIM*patches);


    for(int out_dst_frame = 0; out_dst_frame < dst_frames-1; out_dst_frame++){
      for(unsigned int element_id = First; element_id < Last; element_id++){
        unsigned int in_idx = out_dst_frame*MAX_PATCH_PER_FRAME*DIM + element_id;
        if(dir == 1) in_idx += MAX_PATCH_PER_FRAME*DIM;
        unsigned int out_idx = out_dst_frame*MAX_PATCH_PER_FRAME*DIM + element_id;
        if(dir == 0) out_idx += MAX_PATCH_PER_FRAME*DIM;
        
        Out[out_idx] = In[in_idx];
      }
    }
    
    for(unsigned int element_id = First; element_id < Last; element_id++){
      unsigned out_idx = (dir) ? element_id + (dst_frames-1)*MAX_PATCH_PER_FRAME*DIM : element_id ;
      Out[out_idx] = 0.0;
    }

    return;

}
