/* pfb_gpu.cu 
 * Functions for PFB and FFT in CUDA/GPUs
 */


/* CUDA kernel to multiply all vector elements by 1 (in-place)
 *
 *   Does out_array[i] = in_array[i] * 1 for 0 < i < n_elem.
 *
 */
__global__ void mult_by_1(char *in_array, char* out_array, unsigned nelem)
{
    const int nt = blockDim.x * gridDim.x;
    const int tId = blockIdx.x * blockDim.x + threadIdx.x;

    for (int i = tId; i < nelem; i += nt)
    {
        out_array[i] = in_array[i] * 1;
    }
}


