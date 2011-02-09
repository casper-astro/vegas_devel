/* downsample_gpu.cu
 * Detection/downsampling in GPU/CUDA
 * Paul Demorest, 2009/10
 */
#include <math.h>
#include "dedisperse_gpu.h"
#include "downsample_gpu.h"

/* Returns number of bytes needed for downsampling block */
size_t get_ds_bytes(const struct dedispersion_setup *s) {
    if (s->npol==1) 
        return sizeof(char) * s->npts_per_block / s->dsfac;
    else
        return sizeof(char4) * s->npts_per_block / s->dsfac;
}

/* Initialize the downsampling using values in dedispersion_setup
 * struct.  the s->dsfac field needs to be filled in.
 */
extern "C"
void init_downsample(struct dedispersion_setup *s) {

    // TODO: check that params satisfy any alignment requirements.

    // Allocate memory for DS results on GPU
    const size_t ds_bytes = get_ds_bytes(s);
    cudaMalloc((void**)&s->dsbuf_gpu, ds_bytes);
    printf("Downsample memory = %.1f MB\n", ds_bytes / (1024.*1024.));

    // Check for errors
    cudaThreadSynchronize();
    printf("init_downsample cuda_err='%s'\n", 
            cudaGetErrorString(cudaGetLastError()));
}

/* "naive" version where each thread does one output sample at a time
 * If this isn't fast enough there are lots of optimizations that
 * could be done...
 */
__global__ void detect_downsample_4pol(const float2 *pol0, const float2 *pol1,
        const unsigned dsfac, const unsigned fftlen, const unsigned overlap,
        char4 *out) {

    // Dimensions
    const int tid = threadIdx.x;
    const int nt = blockDim.x;
    const int nvalid = fftlen - overlap;
    const int ifft = blockIdx.x;
    const int iblock = blockIdx.y;
    const int nsamp_per_block = nvalid / gridDim.y;
    const int nout_per_block = nsamp_per_block / dsfac;

    // Data pointers
    const float2 *ptr0 = pol0 + ifft*fftlen + overlap/2 
        + iblock*nsamp_per_block;
    const float2 *ptr1 = pol1 + ifft*fftlen + overlap/2 
        + iblock*nsamp_per_block;
    char4 *optr = out + ifft*nvalid/dsfac + iblock*nout_per_block;

    // Data scaling
    // This should be appropriate for input baseband data with
    // a RMS of ~20 counts.
    const float scale = (float)dsfac * 20.0;

    // Loop over data
    for (int iout=tid; iout<nout_per_block; iout+=nt) {
        float4 otmp= make_float4(0,0,0,0);
        for (int j=0; j<dsfac; j++) {
            float2 p0 = ptr0[iout*dsfac+j];
            float2 p1 = ptr1[iout*dsfac+j];
            otmp.x += p0.x*p0.x + p0.y*p0.y;
            otmp.y += p1.x*p1.x + p1.y*p1.y;
            otmp.z += p0.x*p1.x + p0.y*p1.y;
            otmp.w += p0.x*p1.y - p0.y*p1.x;
        }
        optr[iout].x = __float2int_rn(otmp.x/scale);
        optr[iout].y = __float2int_rn(otmp.y/scale);
        optr[iout].z = __float2int_rn(otmp.z/scale);
        optr[iout].w = __float2int_rn(otmp.w/scale);
    }

}

/* Same as above, except only compute total power */
__global__ void detect_downsample_1pol(const float2 *pol0, const float2 *pol1,
        const unsigned dsfac, const unsigned fftlen, const unsigned overlap,
        char *out) {

    // Dimensions
    const int tid = threadIdx.x;
    const int nt = blockDim.x;
    const int nvalid = fftlen - overlap;
    const int ifft = blockIdx.x;
    const int iblock = blockIdx.y;
    const int nsamp_per_block = nvalid / gridDim.y;
    const int nout_per_block = nsamp_per_block / dsfac;

    // Data pointers
    const float2 *ptr0 = pol0 + ifft*fftlen + overlap/2 
        + iblock*nsamp_per_block;
    const float2 *ptr1 = pol1 + ifft*fftlen + overlap/2 
        + iblock*nsamp_per_block;
    char *optr = out + ifft*nvalid/dsfac + iblock*nout_per_block;

    // Data scaling
    // This should be appropriate for input baseband data with
    // a RMS of ~20 counts in each poln (final 2.0 is for polns).
    const float scale = (float)dsfac * 20.0 * 2.0;

    // Loop over data
    for (int iout=tid; iout<nout_per_block; iout+=nt) {
        float otmp = 0.0;
        for (int j=0; j<dsfac; j++) {
            float2 p0 = ptr0[iout*dsfac+j];
            float2 p1 = ptr1[iout*dsfac+j];
            otmp += p0.x*p0.x + p0.y*p0.y + p1.x*p1.x + p1.y*p1.y;
        }
        optr[iout] = __float2int_rn(otmp/scale);
    }

}


/* Detect / downsample data.  Assumes dedispersion results
 * are already in the GPU, as described in the dedispersion_setup
 * struct.
 */
extern "C"
void downsample(struct dedispersion_setup *s, char *ds_out) {

    /* Sizes */
    const size_t ds_bytes = get_ds_bytes(s);

    /* Benchmark */
#define NT 5
    cudaEvent_t t[NT];
    int it;
    for (it=0; it<NT; it++) cudaEventCreate(&t[it]);
    it=0;

    cudaEventRecord(t[it], 0); it++;
    cudaEventRecord(t[it], 0); it++;

    /* Clear out data buf */
    cudaMemset(s->dsbuf_gpu, 0, ds_bytes);

    /* Downsample data */
    dim3 gd(s->nfft_per_block, 32, 1);
    if (s->npol==1) 
        detect_downsample_1pol<<<gd, 64>>>(s->databuf0_gpu, s->databuf1_gpu,
                s->dsfac, s->fft_len, s->overlap, (char *)s->dsbuf_gpu);
    else
        detect_downsample_4pol<<<gd, 64>>>(s->databuf0_gpu, s->databuf1_gpu,
                s->dsfac, s->fft_len, s->overlap, (char4 *)s->dsbuf_gpu);
    cudaEventRecord(t[it], 0); it++;

    /* Transfer data back to CPU */
    cudaMemcpy(ds_out, s->dsbuf_gpu, ds_bytes, cudaMemcpyDeviceToHost);
    cudaEventRecord(t[it], 0); it++;

    /* Final timer */
    cudaEventRecord(t[it], 0);
    cudaEventSynchronize(t[it]);
    cudaThreadSynchronize();

    /* Add up timers */
    float ttmp;
    it=1;

    cudaEventElapsedTime(&ttmp, t[it], t[it+1]);
    s->time.downsample += ttmp;
    s->time.total2 += ttmp;
    it++;

    cudaEventElapsedTime(&ttmp, t[it], t[it+1]);
    s->time.transfer_to_host += ttmp;
    s->time.total2 += ttmp;
    it++;

    cudaEventElapsedTime(&ttmp, t[0], t[it+1]);
    s->time.total += ttmp;

    /* Cleanup */
    for (it=0; it<NT; it++) cudaEventDestroy(t[it]);

}
