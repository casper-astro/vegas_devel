#ifndef _PFB_GPU_KERNELS
#define _PFB_GPU_KERNELS

/* CUDA kernel to multiply all vector elements by 1 */
__global__ void mult_by_1(char *in_array, char *out_array, unsigned nelem);

#endif
