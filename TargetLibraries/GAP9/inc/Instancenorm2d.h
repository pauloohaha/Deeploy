/*
 * SPDX-FileCopyrightText: 2020 ETH Zurich and University of Bologna
 *
 * SPDX-License-Identifier: Apache-2.0
 * Author: Pu Deng <pudeng@iis.ethz.ch>
 */

#ifndef __INSTANCENORM2D_KERNEL_HEADER_
#define __INSTANCENORM2D_KERNEL_HEADER_

#include "DeeployPULPMath.h"

void PULP_Instancenorm2d_fp32_fp32(float32_t *data_in, float32_t *data_out,
                              float32_t *scale, float32_t *bias,
                              uint32_t size,
                              uint32_t last2DimLength, float32_t epsilon);

#endif // __INSTANCENORM2D_KERNEL_HEADER_