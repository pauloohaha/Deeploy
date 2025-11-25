/*
Author: Pu Deng <pudeng@iis.ethz.ch>
*/

#include "column_softmax_gap9.h"

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

static unsigned short int IntegerExpLUT[] =
{
	0x0001, 0x0002, 0x0007, 0x0014, 0x0036, 0x0094, 0x0193, 0x0448, 0x0BA4, 0x1FA7, 0x560A, 0xE9E2
};

static unsigned short int FractionExpLUT[] =
{
	0x0000, 0x5BF1, 0x31CD, 0x0AF3, 0x4C90, 0x34E2, 0x36E3, 0x510B, 0x7A9F, 0x0ABE, 0x3B9F, 0x1224
};

/* 17.15 fixed point format */
static unsigned short int ExpCoeffLUT[] = {
	0x7FFF, 0x7FFF, 0x4000, 0x1555, 0x0555, 0x0111, 0x002E, 0x0007, 0x0001
};


#define ARRAYSIZE(x)    (sizeof(x) / sizeof(x[ 0 ]))

/* X : fixed point, format Q17.15, returns in Q17.15 */
static unsigned int Exp_fp_17_15(unsigned int X)

{
	int  Result, IntX, FractX, ScaledInt;
  unsigned int Y;
	short int Z_s, FractX_s;
	unsigned short int  ScaledFract;

	if (!X) return 0x8000;
	Y = Abs(X);
	IntX = (Y >> 15);
	/* overflow_mask */
	if (IntX >= ((int) ARRAYSIZE (IntegerExpLUT) - 1)) {
		if (Y==X) return 0x7FFFFFFF; else return 0;
	}
	FractX = (Y & 0x7FFF);
	if (gap_bitextractu(FractX, 1, 14)) {
		/* Taylor series converges quickly only when | FractX | < 0.5 */
		FractX -= 0x8000; IntX++;
	}

	ScaledInt = IntegerExpLUT[IntX]; ScaledFract = FractionExpLUT[IntX];
	/* Taylor's series: exp(x) = 1 + x + x ^ 2 / 2 + x ^ 3 / 3! + x ^ 4 / 4! + x ^ 5 / 5! + x ^ 6 / 6! + x ^ 7 / 7! + x ^ 8 / 8!  */
	FractX_s = FractX; Z_s = FractX; Result = 0;
	for (unsigned int i = 1; i < ARRAYSIZE (ExpCoeffLUT); i++) {
		Result += Z_s*ExpCoeffLUT[i]; // gap_macs(Result, Z, ExpCoeffLUT[ i ]);
		Z_s = gap_mulsRN(Z_s, FractX_s, 15);
	}
	Result = gap_roundnorm(Result, 15) + ExpCoeffLUT[0];
	unsigned short int U_Res = Result;
	Result = gap_muluRN(U_Res, ScaledFract, 15) + U_Res * ScaledInt;
	if (Result && (X > 0x7FFFFFFF)) 
		Result = ((0x7FFFFFFF / Result) >> 1);      /* negative value */
	return (unsigned int) Result;
}

/*  Needs to allocate twice memory for Out for storing the intermediate 16bits results */

void KerColSoftMax8Bits_SQ8 (KerColSoftMax_SQ8_T *Arg)

{
  signed char * __restrict__ In = Arg->In;
	short int * __restrict__ Out = (short int *) Arg->Out;
	int N = Arg->N; //num columns
	int Feat = Arg->Feat; //num rows
	int Norm = Arg->Norm;
  int M[4], Sum[4], InvSum[4];
  unsigned int CoreId = gap_coreid();

  int round = N / (4 * 8); //8 cores, each process 4 element each round, 32 elements per round

  for (int r = 0; r < round; r++) {
    /* Each round of compute */
    int column_id = r * 4 * 8 + CoreId;
    if (column_id >= N) break; //this core has finished alread
    
    /* Find Max */
    for (int i = 0; i < 4; i++)  M[i] = 0x80000000; //reset Max

    for (int row = 0; row < Feat; row++) {
      for (int i = 0; i < 4; i++) {
        M[i] = Max(In[column_id + i + row * N], M[i]);
      }
    }

    /* Compute Exp and find Sum */
    for (int i = 0; i < 4; i++)  Sum[i] = 0; //reset Max
    
    for (int row = 0; row < Feat; row++) {
      for (int i = 0; i < 4; i++) {
        unsigned int Exp = Exp_fp_17_15((In[column_id + i + row * N]-M[i])<<(Norm));
			  Out[column_id + i + row * N] = Exp; 
        Sum[i] += Exp;
      }
    }
    
    /* Compute Inverse and div */
    for (int i = 0; i < 4; i++){
      InvSum[i] = ((FP2FIX(1.0, 15)<<15)/Sum[i]);
    }
    for (int row = 0; row < Feat; row++) {
      for (int i = 0; i < 4; i++) {
        ((signed char *) Out)[column_id + i + row * N] = Abs(gap_roundnorm_reg(Out[column_id + i + row * N]*InvSum[i], 23));
      }
    }

  }

  gap_waitbarrier(0);


}

#define FRAME_ID  0
#define PATCH_PER_FRAME  16
#define MAX_EDGE_PER_PATCH  10
#define DIM   ( 384 )
#define LEN   ( 100 )


void KerColSoftMax_fp32 (void * slave_arg)

{

  KerColSoftMax_fp32_T *Arg = (KerColSoftMax_fp32_T *) slave_arg;
  float * __restrict__ In   = (float *) Arg->In;
	float * __restrict__ Out  = (float *) Arg->Out;
	int N = Arg->N; //num columns
	int Feat = Arg->Feat; //num rows
  float M, Sum, InvSum;
  unsigned int CoreId = gap_coreid();

  int round = (N + 7) / (1 * 8); //8 cores, each process 1 element each round, 8 elements per round, ceiling

  for (int r = 0; r < round; r++) {
    /* Each round of compute */
    int column_id = r * 1 * 8 + CoreId;
    if (column_id >= N) break; //this core has finished alread
    
    /* Find Max */
    M = -MAX_FLT32; //reset Max

    for (int row = 0; row < Feat; row++) {
      M = Maxf32(In[column_id + row * N], M);
    }

    /* Compute Exp and find Sum */
    Sum = 0.0f; //reset Max
    
    for (int row = 0; row < Feat; row++) {
        float Exp = FastExpClampedF32((float)In[column_id + row * N] - M);
			  Out[column_id + row * N] = Exp; 
        Sum += Exp;
    }
    
    /* Compute Inverse and div */
    InvSum = (float)  (1.0F/Sum);

    for (int row = 0; row < Feat; row++) {
        Out[column_id + row * N] = Out[column_id + row * N]*InvSum;
    }

  }

  gap_waitbarrier(0);


}

void SoftMaxAgg_master_kernel(float *L2_net_buffer, int *L2_KK_buffer, float *L2_output_buffer, float *L1_edge_in_buffer, float *L1_edge_out_buffer, int *collected_edge_buffer) {

    for (int patch_id = 0; patch_id < PATCH_PER_FRAME; patch_id++){
        /* process edges from a patch*/
        int patch_kk = FRAME_ID * PATCH_PER_FRAME + patch_id;
        int edge_buff_write_idx = 0; //count how many edges have been written into the buffer

        /* collect edges from the current patch to the L1 buffer*/
        // Piao: TODO: change to DMA
        for (int kk_idx = 0; kk_idx < LEN; kk_idx++){
            if(L2_KK_buffer[kk_idx] == patch_kk){
                /*this edge belongs to the patch being processed*/
                if(edge_buff_write_idx >= MAX_EDGE_PER_PATCH) {
                    printf("L1 egde buffer overflow!\n");
                    return;
                }
                
                collected_edge_buffer[edge_buff_write_idx] = kk_idx;

                for(int copy_id = 0; copy_id < DIM; copy_id++){
                    L1_edge_in_buffer[edge_buff_write_idx*DIM+copy_id] = L2_net_buffer[kk_idx*DIM+copy_id];
                }
                edge_buff_write_idx += 1;//increment the idx after an edge is written into the buffer
            }
        }
        
        /* no edge from the current patch */
        if(edge_buff_write_idx == 0){
           continue;
        }

        KerColSoftMax_fp32_T softmax_arg;
        softmax_arg.In    = L1_edge_in_buffer;
        softmax_arg.Out   = L1_edge_out_buffer;
        softmax_arg.N     = DIM;
        softmax_arg.Feat  = edge_buff_write_idx;

        
        /* dispatch compute workload to the cluster */ 
        pi_cl_team_fork(pi_cl_cluster_nb_cores(), KerColSoftMax_fp32, (void *)&softmax_arg);

        
        /* move the result back to L2 result buffer */
        // Piao: TODO: change to DMA
        for(int edge_id = 0; edge_id < edge_buff_write_idx; edge_id++){
            int egde_L2_buffer_id = collected_edge_buffer[edge_id];
            for(int copy_id = 0; copy_id < DIM; copy_id++){
                L2_output_buffer[egde_L2_buffer_id*DIM+copy_id] = L1_edge_out_buffer[edge_id*DIM+copy_id];
            }
        }
    }

}