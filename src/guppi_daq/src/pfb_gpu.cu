#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cuda.h>
#include <cuda_runtime.h>

#include "guppi_defines.h"
#include "guppi_databuf.h"
#include "pfb_gpu_kernels.h"
#include "spead_heap.h"

/**
 * Global variables: maybe move this to a struct that is passed to each function?
 */
size_t buf_in_block_size;
size_t buf_out_block_size;
size_t buf_index_size;
int nchan;

unsigned char *device_in_buf;
unsigned char *device_out_buf;
struct databuf_index *device_in_index;
struct databuf_index *device_out_index;

/* Initialize all necessary memory, etc for doing PFB 
 * at the given params.
 */
extern "C"
void init_pfb(size_t input_block_sz, size_t output_block_sz, size_t index_sz, int num_chans)
{
	buf_in_block_size = input_block_sz;
    buf_out_block_size = output_block_sz;
	buf_index_size = index_sz;
    nchan = num_chans;

    /* Allocate GPU memory */
    cudaMalloc((void**)&device_in_buf, buf_in_block_size);
    cudaMalloc((void**)&device_out_buf, buf_out_block_size);
    cudaMalloc((void**)&device_in_index, buf_index_size);
    cudaMalloc((void**)&device_out_index, buf_index_size);
}


/* Actually do the PFB by calling CUDA kernels */
extern "C"
void do_pfb(unsigned char *in, unsigned char *out, struct databuf_index *index_in,
                struct databuf_index *index_out)
{
    /* Declare local variables */
    int heap;
    unsigned char *heap_addr_in, *heap_addr_out;
    struct time_spead_heap* time_heap_in;
    struct freq_spead_heap* freq_heap_out;

    /* Copy data block to GPU */
    cudaMemcpy(device_in_buf, in, buf_in_block_size, cudaMemcpyHostToDevice);

    /* Copy block index to GPU */
    cudaMemcpy(device_in_index, index_in, buf_index_size, cudaMemcpyHostToDevice);

    /* Dummy processing */
    int numBlocks = 32;
    int threadsPerBlock = 64;
    gpu_heap_to_cpu_heap<<<numBlocks, threadsPerBlock>>>
            (device_in_buf, device_out_buf, device_in_index, device_out_index, nchan);
     
    /* Transfer data back to host*/
    cudaMemcpy(out, device_out_buf, buf_out_block_size, cudaMemcpyDeviceToHost);

    /* Transfer block index back to host*/
    cudaMemcpy(index_out, device_out_index, buf_index_size, cudaMemcpyDeviceToHost);

    /* Set basic params in output index */
    index_out->num_heaps = index_in->num_heaps;
    index_out->heap_size = sizeof(struct freq_spead_heap) + nchan * 4 * 4;

    /* Write new SPEAD header fields for output heap */
    /* Note: this cannot be done on GPU, due to alignment problems */
    for (heap = 0; heap < index_in->num_heaps; heap++)
    {
        /* Calculate input and output heap addresses */
        heap_addr_in = in + index_in->heap_size*heap;
        time_heap_in = (struct time_spead_heap*)(heap_addr_in);

        heap_addr_out = out + index_out->heap_size*heap;
        freq_heap_out = (struct freq_spead_heap*)(heap_addr_out);
 
        /* Write new heap header fields */
        freq_heap_out->time_cntr_id = 0x20;
        freq_heap_out->time_cntr = time_heap_in->time_cntr;
        freq_heap_out->spectrum_cntr_id = 0x21;
        freq_heap_out->spectrum_cntr = index_in->cpu_gpu_buf[heap].heap_cntr;
        freq_heap_out->integ_size_id = 0x22;
        freq_heap_out->integ_size = 32;
        freq_heap_out->mode_id = 0x23;
        freq_heap_out->mode = time_heap_in->mode;
        freq_heap_out->status_bits_id = 0x24;
        freq_heap_out->status_bits = time_heap_in->status_bits;
        freq_heap_out->payload_data_off_addr_mode = 0x80;
        freq_heap_out->payload_data_off_id = 0x25;
        freq_heap_out->payload_data_off = 0;

        /* Update output index */
        index_out->cpu_gpu_buf[heap].heap_valid =
                 index_in->cpu_gpu_buf[heap].heap_valid;
        index_out->cpu_gpu_buf[heap].heap_cntr =
                index_in->cpu_gpu_buf[heap].heap_cntr;

   }
}

/* 
 * Frees up any allocated memory.
 */
extern "C"
void destroy_pfb()
{
    cudaFree(device_in_buf);
    cudaFree(device_out_buf);
    cudaFree(device_in_index);
    cudaFree(device_out_index);
}