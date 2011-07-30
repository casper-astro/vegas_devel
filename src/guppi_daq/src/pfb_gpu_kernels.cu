/* pfb_gpu.cu 
 * Functions for PFB and FFT in CUDA/GPUs
 */

#include "guppi_defines.h"
#include "guppi_databuf.h"
#include "spead_heap.h"


/*
 * CUDA kernel to convert a GPU heap to a CPU heap.
 * Just simply converts the bytes to floats, and adds the necessary header fields.
 * Each kernel operates on a single, complete heap.
 */
__global__ void gpu_heap_to_cpu_heap(unsigned char *device_in_buf, unsigned char *device_out_buf,
        struct databuf_index* device_in_index, struct databuf_index* device_out_index, int nchan)
{
    /* Determine thread ID */
    const int nt = blockDim.x * gridDim.x;
    const int tId = blockIdx.x * blockDim.x + threadIdx.x;

    /* Calculate output heap size */
    int out_heap_size = sizeof(struct freq_spead_heap) + nchan * 4 * 4;;

    /* Loop through each heap that this kernel must process */
    for (int heap = tId; heap < device_in_index->num_heaps; heap += nt)
    {
        /* Calculate input and output heap addresses */
        unsigned char* heap_addr_in = device_in_buf + device_in_index->heap_size*heap;
        unsigned char *payload_in = (unsigned char*)(heap_addr_in +
                                        sizeof(struct time_spead_heap));

        unsigned char* heap_addr_out = device_out_buf + out_heap_size * heap;
        unsigned int *payload_out = (unsigned int*)(heap_addr_out +
                                        sizeof(struct freq_spead_heap));

        /* Copy heap payload data, converting to unsigned integers */
        for(int samp = 0; (samp < device_in_index->heap_size - sizeof(struct time_spead_heap))
            && (samp < nchan*4); samp++)
        {
            payload_out[samp] = (unsigned int)payload_in[samp];
        }

   }

}

