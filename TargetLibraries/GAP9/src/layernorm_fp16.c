/*
 * SPDX-FileCopyrightText: 2025 ETH Zurich and University of Bologna
 * SPDX-License-Identifier: Apache-2.0
 *
 * FP16 LayerNorm kernel for GAP9.
 * Replicates GAP SDK's KerGlobalAverage1D_fp16 / KerGlobalMeanNormRStd1D_fp16 /
 * KerGlobalMulStd1D_fp16, merged into a single function with epsilon, weight, and bias.
 *
 * Formula: Out = Weight * (In - mean) / sqrt(var + epsilon) + Bias
 */

#include "layernorm_fp16.h"
#include "CNN_Defines_fp16.h"

#ifndef Min
#define Min(a, b) (((a)<(b))?(a):(b))
#endif

void DeeployLayerNorm1D_fp16(KerLayerNorm1D_fp16_T *Arg) {
    F16 * __restrict__ In = Arg->In;
    unsigned int H = Arg->H;
    unsigned int W = Arg->W;
    float * __restrict__ Reduct = Arg->Reduct;
    F16 * __restrict__ Out = Arg->Out;
    F16 * __restrict__ MeanIn = Arg->MeanIn;
    F16 * __restrict__ MeanOut = Arg->MeanOut;
    F16 * __restrict__ Weight = Arg->Weight;
    F16 * __restrict__ Bias = Arg->Bias;
    float Epsilon = Arg->Epsilon;

    unsigned int CoreId = gap_coreid();
    unsigned int NCore = gap_ncore();
    unsigned int Log2Core = gap_fl1(NCore);
    unsigned int Chunk = (W >> Log2Core) + ((W & (NCore - 1)) != 0);
    unsigned int First = Min(Chunk * CoreId, W);
    unsigned int Last = Min(First + Chunk, W);
    unsigned int Iter = Last - First;

    F16 * __restrict__ InRow = In;
    F16 * __restrict__ OutRow = Out;

    for (int Channel = 0; Channel < H; Channel++) {

        /* === Step 1: Compute per-row mean === */
        F16V * __restrict__ vIn = (F16V * __restrict__) &InRow[First];
        F16V vSum = {(F16)0.0f, (F16)0.0f};
        for (unsigned int i = 0; i < Iter / 2; i++) vSum += *vIn++;
        Reduct[CoreId] = (float)(vSum[0] + vSum[1]);
        if (Iter & 0x1) Reduct[CoreId] += (float)*((F16 *)vIn);
        gap_waitbarrier(0);
        if (CoreId == 0) {
            for (unsigned int i = 1; i < NCore; i++) Reduct[0] += Reduct[i];
            MeanOut[Channel] = (F16)(Reduct[0] / W);
        }
        gap_waitbarrier(0);

        /* === Step 2: Center data (Out = In - mean), compute rstd with epsilon === */
        vIn = (F16V * __restrict__) &InRow[First];
        F16V * __restrict__ vOut = (F16V * __restrict__) &OutRow[First];
        F16V vMean = {MeanIn[Channel], MeanIn[Channel]};
        Reduct[CoreId] = 0.0f;
        for (unsigned int i = 0; i < Iter / 2; i++) {
            *vOut = *vIn++ - vMean;
            F16V vSquared = *vOut * *vOut;
            vOut++;
            Reduct[CoreId] += (float)vSquared[0];
            Reduct[CoreId] += (float)vSquared[1];
        }
        if (Iter & 0x1) {
            *((F16 *)vOut) = *((F16 *)vIn) - MeanIn[Channel];
            Reduct[CoreId] += (float)(*((F16 *)vOut) * *((F16 *)vOut));
        }
        gap_waitbarrier(0);
        if (CoreId == 0) {
            for (unsigned int i = 1; i < NCore; i++) Reduct[0] += Reduct[i];
            /* rstd = 1/sqrt(var + eps) = sqrt(W / (sum_sq + eps*W)) */
            MeanOut[Channel] = (F16)Sqrtf32(W / (Reduct[0] + Epsilon * W));
        }
        gap_waitbarrier(0);

        /* === Step 3: Out = centered * rstd * Weight + Bias === */
        F16V * __restrict__ vCentered = (F16V * __restrict__) &OutRow[First];
        vOut = (F16V * __restrict__) &OutRow[First];
        F16V * __restrict__ vWeight = (F16V * __restrict__) &Weight[First];
        F16V * __restrict__ vBias = (F16V * __restrict__) &Bias[First];
        F16V vRstd = {MeanIn[Channel], MeanIn[Channel]};
        for (unsigned int i = 0; i < Iter / 2; i++) {
            *vOut++ = *vCentered++ * vRstd * *vWeight++ + *vBias++;
        }
        if (Iter & 0x1) {
            *((F16 *)vOut) = *((F16 *)vCentered) * MeanIn[Channel] * *((F16 *)vWeight) + *((F16 *)vBias);
        }

        InRow += W;
        OutRow += W;
        gap_waitbarrier(0);
    }
}
