/* pfb_gpu.cu 
 * Functions for PFB and FFT in CUDA/GPUs
 */


/* CUDA kernel to multiply all vector elements by 1 (in-place)
 *
 *   Does array[i] = array[i] * 1 for 0<i<n.
 *
 */
__global__ void mult_by_1(float *array, unsigned nelem)
{
    const int nt = blockDim.x * gridDim.x;
    const int tId = blockIdx.x * blockDim.x + threadIdx.x;

    for (int i = tId; i < nelem; i += nt)
    {
        array[i] = array[i] * 1;
    }
}


