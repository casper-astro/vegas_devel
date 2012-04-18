/* pfb_gpu_kernels.cu 
 * Functions for PFB and FFT in CUDA/GPUs
 */

#include "vegas_defines.h"
#include "vegas_databuf.h"
#include "spead_heap.h"
#include "pfb_gpu.h"


__global__ void DoPFB(char4* pc4Data,
                      float4 *pf4FFTIn,
                      float *pfPFBCoeff)
{
    int i = (blockIdx.x * blockDim.x) + threadIdx.x;
    int iNFFT = (gridDim.x * blockDim.x);
    int j = 0;
    int iAbsIdx = 0;
    float4 f4PFBOut = make_float4(0.0, 0.0, 0.0, 0.0);
    char4 c4Data = make_char4(0, 0, 0, 0);

    for (j = 0; j < VEGAS_NUM_TAPS; ++j)
    {
        /* calculate the absolute index */
        iAbsIdx = (j * iNFFT) + i;
        /* get the address of the block */
        c4Data = pc4Data[iAbsIdx];
        
        f4PFBOut.x += (float) c4Data.x * pfPFBCoeff[iAbsIdx];
        f4PFBOut.y += (float) c4Data.y * pfPFBCoeff[iAbsIdx];
        f4PFBOut.z += (float) c4Data.z * pfPFBCoeff[iAbsIdx];
        f4PFBOut.w += (float) c4Data.w * pfPFBCoeff[iAbsIdx];
    }

    pf4FFTIn[i] = f4PFBOut;

    return;
}

__global__ void Accumulate(float4 *pf4FFTOut,
                           float4 *pf4SumStokes)
{
    int i = (blockIdx.x * blockDim.x) + threadIdx.x;
    float4 f4FFTOut = pf4FFTOut[i];
    float4 f4SumStokes = pf4SumStokes[i];

    /* Re(X)^2 + Im(X)^2 */
    f4SumStokes.x += (f4FFTOut.x * f4FFTOut.x)
                         + (f4FFTOut.y * f4FFTOut.y);
    /* Re(Y)^2 + Im(Y)^2 */
    f4SumStokes.y += (f4FFTOut.z * f4FFTOut.z)
                         + (f4FFTOut.w * f4FFTOut.w);
    /* Re(XY*) */
    f4SumStokes.z += (f4FFTOut.x * f4FFTOut.z)
                         + (f4FFTOut.y * f4FFTOut.w);
    /* Im(XY*) */
    f4SumStokes.w += (f4FFTOut.y * f4FFTOut.z)
                         - (f4FFTOut.x * f4FFTOut.w);

    pf4SumStokes[i] = f4SumStokes;

    return;
}

