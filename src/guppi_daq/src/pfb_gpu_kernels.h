#ifndef _PFB_GPU_KERNELS
#define _PFB_GPU_KERNELS

#include "guppi_defines.h"
#include "guppi_databuf.h"

__global__ void DoPFB(char4* pc4Data,
                      float4 *pf4FFTIn,
                      signed char *pcPFBCoeff);

__global__ void Accumulate(float4 *pf4FFTOut,
                           float4 *pf4SumStokes);

#endif
