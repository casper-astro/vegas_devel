/* pfb_gpu_kernels.cu 
 * Functions for PFB and FFT in CUDA/GPUs
 */

#include "guppi_defines.h"
#include "guppi_databuf.h"
#include "spead_heap.h"
#include "pfb_gpu.h"


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

__global__ void DoPFB(char4* pc4Data,
                      float4* pf4FFTIn)
{
    int i = (blockIdx.x * blockDim.x) + threadIdx.x;
    int iNFFT = (gridDim.x * blockDim.x);
    int j = 0;
    int iAbsIdx = 0;
    float fArg = 0.0;
    float4 f4PFBOut = make_float4(0.0, 0.0, 0.0, 0.0);
    float fCoeff = 0.0;
    char4 c4Data = make_char4(0, 0, 0, 0);

    for (j = 0; j < VEGAS_NUM_TAPS; ++j)
    {
        /* calculate the absolute index */
        iAbsIdx = (j * iNFFT) + i;
        /* get the address of the block */
        c4Data = pc4Data[iAbsIdx];
        /* evaluate filter coefficient at this point */
        fArg = (float) M_PI * ((((float) iAbsIdx) / iNFFT) - (VEGAS_NUM_TAPS / 2));
        fCoeff = (((float) 0.0 == fArg) ? 1.0 : (__sinf(fArg) / fArg));
        
        f4PFBOut.x += (float) c4Data.x * fCoeff;
        f4PFBOut.y += (float) c4Data.y * fCoeff;
        f4PFBOut.z += (float) c4Data.z * fCoeff;
        f4PFBOut.w += (float) c4Data.w * fCoeff;
    }

    pf4FFTIn[i] = f4PFBOut;

    return;
}

__global__ void Accumulate(float4 *pf4FFTOut,
                           float4 *pf4SumStokes,
                           int4 *pi4SumStokes)
{
    int i = (blockIdx.x * blockDim.x) + threadIdx.x;

    /* Re(X)^2 + Im(X)^2 */
    pf4SumStokes[i].x += (pf4FFTOut[i].x * pf4FFTOut[i].x)
                         + (pf4FFTOut[i].y * pf4FFTOut[i].y);
    /* Re(Y)^2 + Im(Y)^2 */
    pf4SumStokes[i].y += (pf4FFTOut[i].z * pf4FFTOut[i].z)
                         + (pf4FFTOut[i].w * pf4FFTOut[i].w);
    /* Re(XY*) */
    pf4SumStokes[i].z += (pf4FFTOut[i].x * pf4FFTOut[i].z)
                         + (pf4FFTOut[i].y * pf4FFTOut[i].w);
    /* Im(XY*) */
    pf4SumStokes[i].w += (pf4FFTOut[i].y * pf4FFTOut[i].z)
                         - (pf4FFTOut[i].x * pf4FFTOut[i].w);


    //temp
    pi4SumStokes[i].x = __float2int_rz(pf4SumStokes[i].x);
    pi4SumStokes[i].y = __float2int_rz(pf4SumStokes[i].y);
    pi4SumStokes[i].z = __float2int_rz(pf4SumStokes[i].z);
    pi4SumStokes[i].w = __float2int_rz(pf4SumStokes[i].w);

    return;
}

__global__ void Convert(float4 *pf4SumStokes,
                        int4 *pi4SumStokes)
{
    int i = (blockIdx.x * blockDim.x) + threadIdx.x;

    pi4SumStokes[i].x = __float2int_rz(pf4SumStokes[i].x);
    pi4SumStokes[i].y = __float2int_rz(pf4SumStokes[i].y);
    pi4SumStokes[i].z = __float2int_rz(pf4SumStokes[i].z);
    pi4SumStokes[i].w = __float2int_rz(pf4SumStokes[i].w);

    return;
}

