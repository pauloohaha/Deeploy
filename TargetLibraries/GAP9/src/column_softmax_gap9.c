/*
Author: Pu Deng <pudeng@iis.ethz.ch>
*/

#include "column_softmax_gap9.h"

//#define PERF_PROFILE

static int CoreCountDynamic = 1;
static int ActiveCore = gap_ncore();

#ifdef PERF_PROFILE
static PI_L2 uint32_t CoreActiveCnt[8];
#endif

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

void KerColSoftMax_fp32 (void * slave_arg)

{  
  KerColSoftMax_fp32_T *Arg = (KerColSoftMax_fp32_T *) slave_arg;
  float * __restrict__ In   = (float *) Arg->In;
	float * __restrict__ Out  = (float *) Arg->Out;
	int N = Arg->N; //num columns
	int Feat = Arg->Feat; //num rows
  float M, Sum, InvSum;
  unsigned int CoreId = gap_coreid();

#ifdef PERF_PROFILE
  pi_perf_conf( 1 << PI_PERF_ACTIVE_CYCLES);
  pi_perf_reset();
  pi_perf_start();
#endif

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

#ifdef PERF_PROFILE
  pi_perf_stop();
  CoreActiveCnt[CoreId] = pi_perf_read(PI_PERF_ACTIVE_CYCLES);
#endif

}


void KerColSum_fp32 (void * slave_arg)

{  
  KerColSoftMax_fp32_T *Arg = (KerColSoftMax_fp32_T *) slave_arg;
  float * __restrict__ In   = (float *) Arg->In;
	float * __restrict__ Out  = (float *) Arg->Out;
	int N = Arg->N; //num columns
	int Feat = Arg->Feat; //num rows
  unsigned int CoreId = gap_coreid();

#ifdef PERF_PROFILE
  pi_perf_conf( 1 << PI_PERF_ACTIVE_CYCLES);
  pi_perf_reset();
  pi_perf_start();
#endif

  int round = (N + 7) / (1 * 8); //8 cores, each process 1 element each round, 8 elements per round, ceiling

  for (int r = 0; r < round; r++) {
    /* Each round of compute */
    int column_id = r * 1 * 8 + CoreId;
    if (column_id >= N) break; //this core has finished alread
    
    /* Find Sum */
    Out[column_id] = In[column_id + 0 * N];

    for (int row = 1; row < Feat; row++) {
      Out[column_id] += In[column_id + row * N];
    }
  }

  gap_waitbarrier(0);

#ifdef PERF_PROFILE
  pi_perf_stop();
  CoreActiveCnt[CoreId] = pi_perf_read(PI_PERF_ACTIVE_CYCLES);
#endif

}


void KerCatter_fp32 (void * slave_arg)

{  
  KerColScatter_fp32_T *Arg = (KerColScatter_fp32_T *) slave_arg;
  float * __restrict__ In   = (float *) Arg->In;
  float* __restrict__ In_agg = (float *) Arg->In_agg;
	int N = Arg->N; //num columns
	int Feat = Arg->Feat; //num rows
  float M, Sum, InvSum;
  unsigned int CoreId = gap_coreid();

#ifdef PERF_PROFILE
  pi_perf_conf( 1 << PI_PERF_ACTIVE_CYCLES);
  pi_perf_reset();
  pi_perf_start();
#endif

  int round = (N + 7) / (1 * 8); //8 cores, each process 1 element each round, 8 elements per round, ceiling

  for (int r = 0; r < round; r++) {
    /* Each round of compute */
    int column_id = r * 1 * 8 + CoreId;
    if (column_id >= N) break; //this core has finished alread
    
    /* scatter add */
    for (int row = 0; row < Feat; row++) {
      In[column_id + row * N] += In_agg[column_id];
    }
  }

  gap_waitbarrier(0);

#ifdef PERF_PROFILE
  pi_perf_stop();
  CoreActiveCnt[CoreId] = pi_perf_read(PI_PERF_ACTIVE_CYCLES);
#endif

}

void ColSoftMax_master_kernel(float *L2_net_buffer, int *L2_KK_buffer, float *L2_output_buffer, 
                              float *L1_edge_in_ping_buffer, float *L1_edge_out_ping_buffer, 
                              float *L1_edge_in_pong_buffer, float *L1_edge_out_pong_buffer,
                              int dir // 0 for patch softmax agg, 1 for frame softmax agg
                               ) {

    pi_cl_dma_cmd_t data_in_ping_dma_handle; //maximum transfer MAX_EDGE_PER_PATCH edges
    pi_cl_dma_cmd_t data_in_pong_dma_handle; 
    pi_cl_dma_cmd_t data_out_ping_dma_handle; 
    pi_cl_dma_cmd_t data_out_pong_dma_handle; 

    /*Ping Pong array*/
    float *L1_edge_in_buffer[2]   = {L1_edge_in_ping_buffer, L1_edge_in_pong_buffer};
    float *L1_edge_out_buffer[2]  = {L1_edge_out_ping_buffer, L1_edge_out_pong_buffer};
    pi_cl_dma_cmd_t *data_in_dma_handles[2]  = {&data_in_ping_dma_handle, &data_in_pong_dma_handle};
    pi_cl_dma_cmd_t *data_out_dma_handles[2] = {&data_out_ping_dma_handle, &data_out_pong_dma_handle};

    /* Initial Ping buffer setup*/
    int num_patches     = L2_KK_buffer[0];
    int num_dst_frames  = L2_KK_buffer[1];

    if(num_dst_frames == 0){
        /* no dst frame */
        return;
    }
    
    /* issue the 2d data transfer in 1 shot */
    if(dir == 0){
      //patch agg, transfer a column in
      pi_cl_dma_cmd_2d((uint32_t)&L2_net_buffer[0*DIM], (uint32_t)L1_edge_in_buffer[0], //ext addr, loc addr, 
                        num_dst_frames * (uint32_t)DIM * sizeof(float),         // total size
                        (uint32_t)DIM * sizeof(float) * MAX_PATCH_PER_FRAME,    // 2d stride
                        (uint32_t)DIM * sizeof(float),                          // length per section
                        PI_CL_DMA_DIR_EXT2LOC,  data_in_dma_handles[0]);
    } else {
      //frame agg, transfer a row in
      pi_cl_dma_cmd_2d((uint32_t)&L2_net_buffer[0*DIM], (uint32_t)L1_edge_in_buffer[0], //ext addr, loc addr, 
                        num_patches * (uint32_t)DIM * sizeof(float),         // total size
                        (uint32_t)DIM * sizeof(float),                          // 2d stride
                        (uint32_t)DIM * sizeof(float),                          // length per section
                        PI_CL_DMA_DIR_EXT2LOC,  data_in_dma_handles[0]);
    }

    int total_iterations;
    if(dir == 0){
      total_iterations = num_patches;
    } else {
      total_iterations = num_dst_frames;
    }

    int softmax_feat;
    if(dir == 0){
      softmax_feat  = num_dst_frames;
    } else {
      softmax_feat  = num_patches;
    }

    for (int iteration_id = 0; iteration_id < total_iterations; iteration_id++){
#ifdef PERF_PROFILE
        for (int i = 0; i < 8; i++){
          CoreActiveCnt[i] = 0;
        }
        pi_perf_conf(1 << PI_PERF_CYCLES | 1 << PI_PERF_ACTIVE_CYCLES);
        pi_perf_reset();
        pi_perf_start();
#endif
        int compute_bin = iteration_id % 2; //0 for currently computing Ping, 1 for currently computing Pong 
        int data_bin    = (iteration_id+1) % 2; //the buffer that is currently moving data

        /* collect edges from the current patch to the L1 buffer */
        /* set up data in transfer for the next iteration, except for the last iteration */
        if(iteration_id < total_iterations - 1){
          if(dir == 0){
            //patch agg, transfer a column in
            pi_cl_dma_cmd_2d((uint32_t)&L2_net_buffer[(iteration_id+1)*DIM], (uint32_t)L1_edge_in_buffer[data_bin], //ext addr, loc addr
                          num_dst_frames * (uint32_t)DIM * sizeof(float),                     // total size
                          (uint32_t)DIM * sizeof(float) * MAX_PATCH_PER_FRAME,                // 2d stride
                          (uint32_t)DIM * sizeof(float),                                      // length per section
                          PI_CL_DMA_DIR_EXT2LOC,  data_in_dma_handles[data_bin]);
          } else {
            //frame agg, transfer a row in
            pi_cl_dma_cmd_2d((uint32_t)&L2_net_buffer[(iteration_id+1)*DIM*MAX_PATCH_PER_FRAME], (uint32_t)L1_edge_in_buffer[data_bin], //ext addr, loc addr
                          num_patches * (uint32_t)DIM * sizeof(float),                     // total size
                          (uint32_t)DIM * sizeof(float),                                      // 2d stride
                          (uint32_t)DIM * sizeof(float),                                      // length per section
                          PI_CL_DMA_DIR_EXT2LOC,  data_in_dma_handles[data_bin]);
          }
        }

        /* wait for the input data dma finish before start compute */
        pi_cl_dma_cmd_wait(data_in_dma_handles[compute_bin]);

        /* wait for the output data dma two iterations before to finis */
        if(iteration_id >= 2){
          /*only need to start checking previous output dma transfer starting 3rd iteration*/
          pi_cl_dma_cmd_wait(data_out_dma_handles[compute_bin]);
        }
       
        KerColSoftMax_fp32_T softmax_arg;
        softmax_arg.In    = L1_edge_in_buffer[compute_bin];
        softmax_arg.Out   = L1_edge_out_buffer[compute_bin];
        softmax_arg.N     = DIM;
        softmax_arg.Feat  = softmax_feat;

        /* dispatch compute workload to the cluster */ 
        pi_cl_team_fork(pi_cl_cluster_nb_cores(), KerColSoftMax_fp32, (void *)&softmax_arg);

        /* move the result back to L2 result buffer for the current iteration */
        if(dir == 0){
          pi_cl_dma_cmd_2d((uint32_t)&L2_output_buffer[iteration_id*DIM], (uint32_t)L1_edge_out_buffer[compute_bin], //ext addr, loc addr, 
                      num_dst_frames * (uint32_t)DIM * sizeof(float),                 //total size
                      (uint32_t)DIM * sizeof(float) * MAX_PATCH_PER_FRAME,            // 2d stride
                      (uint32_t)DIM * sizeof(float),                                  // length per section
                      PI_CL_DMA_DIR_LOC2EXT,  data_out_dma_handles[compute_bin]);
        } else {
          pi_cl_dma_cmd_2d((uint32_t)&L2_output_buffer[iteration_id*DIM*MAX_PATCH_PER_FRAME], (uint32_t)L1_edge_out_buffer[compute_bin], //ext addr, loc addr, 
                      num_patches * (uint32_t)DIM * sizeof(float),                 //total size
                      (uint32_t)DIM * sizeof(float),                                  // 2d stride
                      (uint32_t)DIM * sizeof(float),                                  // length per section
                      PI_CL_DMA_DIR_LOC2EXT,  data_out_dma_handles[compute_bin]);
        }
        if(iteration_id == total_iterations-1){
          /*wait for last DMA output*/
          pi_cl_dma_cmd_wait(data_out_dma_handles[compute_bin]);
          if(total_iterations >= 2){
            /* free the other output dma */
            pi_cl_dma_cmd_wait(data_out_dma_handles[data_bin]);
          }
        }
#ifdef PERF_PROFILE
        pi_perf_stop();
        uint32_t cycles = pi_perf_read(PI_PERF_ACTIVE_CYCLES);
        uint32_t tim_cycles = pi_perf_read(PI_PERF_CYCLES);
        printf("Perf : %d cycles Timer : %d cycles, Feat:%d\n", cycles, tim_cycles, softmax_feat);
        for (int i = 0; i < 8; i++){
          printf("Core: %d, active cycle:%d\n", i, CoreActiveCnt[i]);
        }
#endif

    }

}


/* perfrom column sum across L2_net buffer */
void ColSum_master_kernel(float *L2_net_buffer, int *L2_KK_buffer, float *L2_output_buffer, 
                              float *L1_edge_in_ping_buffer, float *L1_edge_out_ping_buffer, 
                              float *L1_edge_in_pong_buffer, float *L1_edge_out_pong_buffer,
                              int dir) {
    pi_cl_dma_cmd_t data_in_ping_dma_handle; //maximum transfer MAX_EDGE_PER_PATCH edges
    pi_cl_dma_cmd_t data_in_pong_dma_handle; 
    pi_cl_dma_cmd_t data_out_ping_dma_handle; 
    pi_cl_dma_cmd_t data_out_pong_dma_handle; 

    /*Ping Pong array*/
    float *L1_edge_in_buffer[2]   = {L1_edge_in_ping_buffer, L1_edge_in_pong_buffer};
    float *L1_edge_out_buffer[2]  = {L1_edge_out_ping_buffer, L1_edge_out_pong_buffer};
    pi_cl_dma_cmd_t *data_in_dma_handles[2]  = {&data_in_ping_dma_handle, &data_in_pong_dma_handle};
    pi_cl_dma_cmd_t *data_out_dma_handles[2] = {&data_out_ping_dma_handle, &data_out_pong_dma_handle};

    /* Initial Ping buffer setup*/
    int num_patches     = L2_KK_buffer[0];
    int num_dst_frames  = L2_KK_buffer[1];

    if(num_dst_frames == 0){
        /* no dst frame */
        return;
    }
    
    /* issue the 2d data transfer in 1 shot */
    if(dir == 0){
      //patch agg, transfer a column in
      pi_cl_dma_cmd_2d((uint32_t)&L2_net_buffer[0*DIM], (uint32_t)L1_edge_in_buffer[0], //ext addr, loc addr, 
                        num_dst_frames * (uint32_t)DIM * sizeof(float),         // total size
                        (uint32_t)DIM * sizeof(float) * MAX_PATCH_PER_FRAME,    // 2d stride
                        (uint32_t)DIM * sizeof(float),                          // length per section
                        PI_CL_DMA_DIR_EXT2LOC,  data_in_dma_handles[0]);
    } else {
      //frame agg, transfer a row in
      pi_cl_dma_cmd_2d((uint32_t)&L2_net_buffer[0*DIM], (uint32_t)L1_edge_in_buffer[0], //ext addr, loc addr, 
                        num_patches * (uint32_t)DIM * sizeof(float),         // total size
                        (uint32_t)DIM * sizeof(float),                          // 2d stride
                        (uint32_t)DIM * sizeof(float),                          // length per section
                        PI_CL_DMA_DIR_EXT2LOC,  data_in_dma_handles[0]);
    }

    int total_iterations;
    if(dir == 0){
      total_iterations = num_patches;
    } else {
      total_iterations = num_dst_frames;
    }

    int softmax_feat;
    if(dir == 0){
      softmax_feat  = num_dst_frames;
    } else {
      softmax_feat  = num_patches;
    }

    for (int iteration_id = 0; iteration_id < total_iterations; iteration_id++){
#ifdef PERF_PROFILE
        for (int i = 0; i < 8; i++){
          CoreActiveCnt[i] = 0;
        }
        pi_perf_conf(1 << PI_PERF_CYCLES | 1 << PI_PERF_ACTIVE_CYCLES);
        pi_perf_reset();
        pi_perf_start();
#endif
        int compute_bin = iteration_id % 2; //0 for currently computing Ping, 1 for currently computing Pong 
        int data_bin    = (iteration_id+1) % 2; //the buffer that is currently moving data

        /* collect edges from the current patch to the L1 buffer */
        /* set up data in transfer for the next iteration, except for the last iteration */
        if(iteration_id < total_iterations - 1){
          if(dir == 0){
            //patch agg, transfer a column in
            pi_cl_dma_cmd_2d((uint32_t)&L2_net_buffer[(iteration_id+1)*DIM], (uint32_t)L1_edge_in_buffer[data_bin], //ext addr, loc addr
                          num_dst_frames * (uint32_t)DIM * sizeof(float),                     // total size
                          (uint32_t)DIM * sizeof(float) * MAX_PATCH_PER_FRAME,                // 2d stride
                          (uint32_t)DIM * sizeof(float),                                      // length per section
                          PI_CL_DMA_DIR_EXT2LOC,  data_in_dma_handles[data_bin]);
          } else {
            //frame agg, transfer a row in
            pi_cl_dma_cmd_2d((uint32_t)&L2_net_buffer[(iteration_id+1)*DIM*MAX_PATCH_PER_FRAME], (uint32_t)L1_edge_in_buffer[data_bin], //ext addr, loc addr
                          num_patches * (uint32_t)DIM * sizeof(float),                     // total size
                          (uint32_t)DIM * sizeof(float),                                      // 2d stride
                          (uint32_t)DIM * sizeof(float),                                      // length per section
                          PI_CL_DMA_DIR_EXT2LOC,  data_in_dma_handles[data_bin]);
          }
        }

        /* wait for the input data dma finish before start compute */
        pi_cl_dma_cmd_wait(data_in_dma_handles[compute_bin]);

        /* wait for the output data dma two iterations before to finis */
        if(iteration_id >= 2){
          /*only need to start checking previous output dma transfer starting 3rd iteration*/
          pi_cl_dma_cmd_wait(data_out_dma_handles[compute_bin]);
        }
       
        KerColSoftMax_fp32_T softmax_arg;
        softmax_arg.In    = L1_edge_in_buffer[compute_bin];
        softmax_arg.Out   = L1_edge_out_buffer[compute_bin];
        softmax_arg.N     = DIM;
        softmax_arg.Feat  = softmax_feat;

        /* dispatch compute workload to the cluster */ 
        pi_cl_team_fork(pi_cl_cluster_nb_cores(), KerColSum_fp32, (void *)&softmax_arg);

        /* move the result back to L2 result buffer for the current iteration */
        // in column sum case, the output is always consecutive total_iterations of DIM vector, thus same for both patch agg or frame agg
        pi_cl_dma_cmd_2d((uint32_t)&L2_output_buffer[iteration_id*DIM], (uint32_t)L1_edge_out_buffer[compute_bin], //ext addr, loc addr, 
                      (uint32_t)DIM * sizeof(float),                                  //total size
                      (uint32_t)DIM * sizeof(float),                                  // 2d stride
                      (uint32_t)DIM * sizeof(float),                                  // length per section
                      PI_CL_DMA_DIR_LOC2EXT,  data_out_dma_handles[compute_bin]);

        if(iteration_id == total_iterations-1){
          /*wait for last DMA output*/
          pi_cl_dma_cmd_wait(data_out_dma_handles[compute_bin]);
          if(total_iterations >= 2){
            /* free the other output dma */
            pi_cl_dma_cmd_wait(data_out_dma_handles[data_bin]);
          }
        }
#ifdef PERF_PROFILE
        pi_perf_stop();
        uint32_t cycles = pi_perf_read(PI_PERF_ACTIVE_CYCLES);
        uint32_t tim_cycles = pi_perf_read(PI_PERF_CYCLES);
        printf("Perf : %d cycles Timer : %d cycles, Feat:%d\n", cycles, tim_cycles, softmax_feat);
        for (int i = 0; i < 8; i++){
          printf("Core: %d, active cycle:%d\n", i, CoreActiveCnt[i]);
        }
#endif

    }

}

/* add each edge in L2_agg buffer to a column of edges in L2_net, L1 inplace add */
void ColScatter_master_kernel(float *L2_net_buffer, int *L2_KK_buffer, float* L2_agg_buffer, float *L2_output_buffer, 
                              float *L1_edge_ping_buffer, float *L1_agg_ping_buffer,
                              float *L1_edge_pong_buffer, float *L1_agg_pong_buffer,
                              int dir) {
    
    pi_cl_dma_cmd_t data_in_ping_dma_handle; //maximum transfer MAX_EDGE_PER_PATCH edges
    pi_cl_dma_cmd_t data_in_pong_dma_handle; 
    pi_cl_dma_cmd_t agg_in_ping_dma_handle; 
    pi_cl_dma_cmd_t agg_in_pong_dma_handle; 
    pi_cl_dma_cmd_t data_out_ping_dma_handle; 
    pi_cl_dma_cmd_t data_out_pong_dma_handle; 

    /*Ping Pong array*/
    float *L1_edge_buffer[2]      = {L1_edge_ping_buffer, L1_edge_pong_buffer};
    float *L1_agg_in_buffer[2]    = {L1_agg_ping_buffer, L1_agg_pong_buffer};
    pi_cl_dma_cmd_t *data_in_dma_handles[2]  = {&data_in_ping_dma_handle, &data_in_pong_dma_handle};
    pi_cl_dma_cmd_t *agg_in_dma_handles[2]  = {&agg_in_ping_dma_handle, &agg_in_pong_dma_handle};
    pi_cl_dma_cmd_t *data_out_dma_handles[2] = {&data_out_ping_dma_handle, &data_out_pong_dma_handle};

    /* Initial Ping buffer setup*/
    int num_patches     = L2_KK_buffer[0];
    int num_dst_frames  = L2_KK_buffer[1];

    if(num_dst_frames == 0 || num_patches == 0){
        /* no dst frame */
        return;
    }
    
    /* issue the 2d data transfer in 1 shot */
    /* in net dma */
    if(dir == 0){
      pi_cl_dma_cmd_2d((uint32_t)&L2_net_buffer[0*DIM], (uint32_t)L1_edge_buffer[0], //ext addr, loc addr, 
                        num_dst_frames * (uint32_t)DIM * sizeof(float),         // total size
                        (uint32_t)DIM * sizeof(float) * MAX_PATCH_PER_FRAME,    // 2d stride
                        (uint32_t)DIM * sizeof(float),                          // length per section
                        PI_CL_DMA_DIR_EXT2LOC,  data_in_dma_handles[0]);
    } else {
      pi_cl_dma_cmd_2d((uint32_t)&L2_net_buffer[0*DIM], (uint32_t)L1_edge_buffer[0], //ext addr, loc addr, 
                        num_patches * (uint32_t)DIM * sizeof(float),            // total size
                        (uint32_t)DIM * sizeof(float),                          // 2d stride
                        (uint32_t)DIM * sizeof(float),                          // length per section
                        PI_CL_DMA_DIR_EXT2LOC,  data_in_dma_handles[0]);
    }
    
    /* in agg dma */
    pi_cl_dma_cmd((uint32_t)&L2_agg_buffer[0*DIM], (uint32_t)L1_agg_in_buffer[0],
                (uint32_t)DIM * sizeof(float), PI_CL_DMA_DIR_EXT2LOC, agg_in_dma_handles[0]);

    int total_iterations;
    if(dir == 0){
      total_iterations = num_patches;
    } else {
      total_iterations = num_dst_frames;
    }

    int softmax_feat;
    if(dir == 0){
      softmax_feat  = num_dst_frames;
    } else {
      softmax_feat  = num_patches;
    }


    for (int iteration_id = 0; iteration_id < total_iterations; iteration_id++){

#ifdef PERF_PROFILE
        for (int i = 0; i < 8; i++){
          CoreActiveCnt[i] = 0;
        }
        pi_perf_conf(1 << PI_PERF_CYCLES | 1 << PI_PERF_ACTIVE_CYCLES);
        pi_perf_reset();
        pi_perf_start();
#endif
        int compute_bin = iteration_id % 2; //0 for currently computing Ping, 1 for currently computing Pong 
        int data_bin    = (iteration_id+1) % 2; //the buffer that is currently moving data

        /* collect edges from the current patch to the L1 buffer */
        /* set up data in transfer for the next iteration, except for the last iteration */
        if(iteration_id < total_iterations - 1){
            /* in net dma */
            if(dir == 0){
              // transfer a column in
              pi_cl_dma_cmd_2d((uint32_t)&L2_net_buffer[(iteration_id+1)*DIM], (uint32_t)L1_edge_buffer[data_bin], //ext addr, loc addr
                          num_dst_frames * (uint32_t)DIM * sizeof(float),                     // total size
                          (uint32_t)DIM * sizeof(float) * MAX_PATCH_PER_FRAME,                // 2d stride
                          (uint32_t)DIM * sizeof(float),                                      // length per section
                          PI_CL_DMA_DIR_EXT2LOC,  data_in_dma_handles[data_bin]);
            } else {
              // transfer a row in
              pi_cl_dma_cmd_2d((uint32_t)&L2_net_buffer[(iteration_id+1)*DIM*MAX_PATCH_PER_FRAME], (uint32_t)L1_edge_buffer[data_bin], //ext addr, loc addr
                          num_patches * (uint32_t)DIM * sizeof(float),                        // total size
                          (uint32_t)DIM * sizeof(float),                                      // 2d stride
                          (uint32_t)DIM * sizeof(float),                                      // length per section
                          PI_CL_DMA_DIR_EXT2LOC,  data_in_dma_handles[data_bin]);
            }

            /* in agg dma */
            // in agg data is always consecutive total iteration * DIM vector, same for both frame agg and patch agg
            pi_cl_dma_cmd((uint32_t)&L2_agg_buffer[(iteration_id+1)*DIM], (uint32_t)L1_agg_in_buffer[data_bin],
                          (uint32_t)DIM * sizeof(float), PI_CL_DMA_DIR_EXT2LOC, agg_in_dma_handles[data_bin]);
        }

        /* wait for the input data dma finish before start compute */
        pi_cl_dma_cmd_wait(data_in_dma_handles[compute_bin]);
        pi_cl_dma_cmd_wait(agg_in_dma_handles[compute_bin]);

        /* wait for the output data dma two iterations before to finis */
        if(iteration_id >= 2){
          /*only need to start checking previous output dma transfer starting 3rd iteration*/
          pi_cl_dma_cmd_wait(data_out_dma_handles[compute_bin]);
        }
       
        KerColScatter_fp32_T scatter_arg;
        scatter_arg.In      = L1_edge_buffer[compute_bin];
        scatter_arg.In_agg  = L1_agg_in_buffer[compute_bin];
        scatter_arg.N       = DIM;
        scatter_arg.Feat    = softmax_feat;

        /* dispatch compute workload to the cluster */ 
        pi_cl_team_fork(pi_cl_cluster_nb_cores(), KerCatter_fp32, (void *)&scatter_arg);

        /* move the result back to L2 result buffer for the current iteration */
        if(dir == 0){
          pi_cl_dma_cmd_2d((uint32_t)&L2_output_buffer[iteration_id*DIM], (uint32_t)L1_edge_buffer[compute_bin], //ext addr, loc addr, 
                      num_dst_frames * (uint32_t)DIM * sizeof(float),                 // total size
                      (uint32_t)DIM * sizeof(float) * MAX_PATCH_PER_FRAME,            // 2d stride
                      (uint32_t)DIM * sizeof(float),                                  // length per section
                      PI_CL_DMA_DIR_LOC2EXT,  data_out_dma_handles[compute_bin]);
        } else {
          pi_cl_dma_cmd_2d((uint32_t)&L2_output_buffer[iteration_id*DIM*MAX_PATCH_PER_FRAME], (uint32_t)L1_edge_buffer[compute_bin], //ext addr, loc addr, 
                      num_patches * (uint32_t)DIM * sizeof(float),                    // total size
                      (uint32_t)DIM * sizeof(float),                                  // 2d stride
                      (uint32_t)DIM * sizeof(float),                                  // length per section
                      PI_CL_DMA_DIR_LOC2EXT,  data_out_dma_handles[compute_bin]);
        }
#ifdef PERF_PROFILE
        pi_perf_stop();
        uint32_t cycles = pi_perf_read(PI_PERF_ACTIVE_CYCLES);
        uint32_t tim_cycles = pi_perf_read(PI_PERF_CYCLES);
        printf("Perf : %d cycles Timer : %d cycles, Feat:%d\n", cycles, tim_cycles, num_dst_frames);
        for (int i = 0; i < 8; i++){
          printf("Core: %d, active cycle:%d\n", i, CoreActiveCnt[i]);
        }
#endif
        if(iteration_id == total_iterations-1){
          /*wait for last DMA output*/
          pi_cl_dma_cmd_wait(data_out_dma_handles[compute_bin]);
          if(total_iterations >= 2){
            /* free the other output dma */
            pi_cl_dma_cmd_wait(data_out_dma_handles[data_bin]);
          }
        }


    }


}