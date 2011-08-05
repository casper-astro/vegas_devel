#ifndef _PFB_GPU_KERNELS
#define _PFB_GPU_KERNELS

#include "guppi_defines.h"
#include "guppi_databuf.h"

/* CUDA kernel to multiply all vector elements by 1 and convert to float */
__global__ void gpu_heap_to_cpu_heap(unsigned char *device_in_buf, unsigned char *device_out_buf,
        struct databuf_index* device_in_index, struct databuf_index* device_out_index, int nchan);

__global__ void DoPFB(char4* pc4Data,
                      float4* pf4FFTIn);

__global__ void Accumulate(float4 *pf4FFTOut,
                           float4 *pf4SumStokes,
                           int4 *pi4SumStokes);

__global__ void Convert(float4 *pf4SumStokes,
                        int4 *pi4SumStokes);

#endif
