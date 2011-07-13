#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>


/**
 * Global variables: maybe move this to a struct that is passed to each function?
 */
size_t buf_block_size;
size_t buf_index_size;

char *device_in_buf;
char *device_out_buf;


/* Initialize all necessary memory, etc for doing PFB 
 * at the given params.
 */
void init_pfb(size_t block_size, size_t index_size)
{
	buf_block_size = block_size;
	buf_index_size = index_size;

    /* Allocate GPU memory */
    cudaMalloc((void**)&device_in_buf, buf_block_size);
    cudaMalloc((void**)&device_out_buf, buf_block_size);
}


/* Actually do the PFB by calling CUDA kernels */
void do_pfb(const char *in, char *out, struct databuf_index* index_in,
                struct databuf_index* index_out)
{
    /* Copy data to GPU */
    cudaMemcpy(device_in_buf, in, block_size, cudaMemcpyHostToDevice);

    /* Dummy processing */
    int numBlocks = 32;
    int threadsPerBlock = 64;
    mult_by_1<<<numBlocks, threadsPerBlock>>>(in, out, buf_block_size);
     
    /* Transfer data back to host*/
    cudaMemcpy(out, device_out_buf, block_size, cudaMemcpyDeviceToHost);

    /* Copy input index to output index: for initial testing only */
	memcpy(index_out, index_in, index_size);
}
