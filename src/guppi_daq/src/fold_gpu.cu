/* fold_gpu.cu
 * Folding in CUDA/GPUs
 * Paul Demorest, 2009/09
 */
#include <math.h>
#include "fold.h"
#include "fold_gpu.h"
#include "dedisperse_gpu.h"
#include "polyco.h"

//const int thread_per_block = 32;

/* Initializes info needed for folding in the GPU.
 * init_dedispersion() must have already been called
 * using the input struct.
 * s->nbins_fold needs to be set before calling this function.
 */
extern "C"
void init_fold(struct dedispersion_setup *s) {

    // Try to go for ~1024 fold threads
    //s->fold_thread_per_block = 1;
    //while (s->fold_thread_per_block * s->nfft_per_block < 1024) 
    //    s->fold_thread_per_block <<= 1;
    //if (s->fold_thread_per_block > 512) s->fold_thread_per_block = 512;
    //printf("fold_thread_per_block = %d\n", s->fold_thread_per_block);
    s->fold_thread_per_block = 64;

    // Make sure fft_len and overlap agree with threads_per_block
    if ((s->fft_len - s->overlap) % 64) {
        printf("Warning, nvalid=%d does not divide evenly into threads_per_block=%d\n", 
                s->fft_len - s->overlap, s->fold_thread_per_block);
    }

    // These hold fold phase and freq info to be used by gpu
    cudaMalloc((void**)&s->fold_phase, sizeof(double) * s->nfft_per_block);
    cudaMalloc((void**)&s->fold_step, sizeof(double) * s->nfft_per_block);

    // These hold data
    const unsigned npol = 4;  // Assume always full-Stokes for now..
    const size_t bytes_per_profile = sizeof(float) * npol * s->nbins_fold;
    const size_t bytes_per_count = sizeof(unsigned) * s->nbins_fold;
    cudaMalloc((void**)&s->foldtmp_gpu, bytes_per_profile * s->nfft_per_block);
    cudaMalloc((void**)&s->foldbuf_gpu, bytes_per_profile);
    cudaMalloc((void**)&s->foldtmp_c_gpu, bytes_per_count * s->nfft_per_block);
    cudaMalloc((void**)&s->foldbuf_c_gpu, bytes_per_count);

    printf("Fold nbin = %d\n", s->nbins_fold);
    printf("Fold memory = %.1f MB\n", ((bytes_per_profile+bytes_per_count) * 
            (s->nfft_per_block + 1)) /(1024.*1024.));

    // Check for alloc error
    cudaThreadSynchronize();
    printf("init_fold cuda_err=\'%s\'\n", cudaGetErrorString(cudaGetLastError()));
}

/* Compute fold phases and freqs for the given MJD and transfer them
 * to the GPU.  MJD refers to the midpoint time of the first sample
 * in the data block.
 */
extern "C"
int compute_fold_params(struct dedispersion_setup *s, const struct polyco *pc) {

    /* Check that MJD is in polyco range */
    if (pc_out_of_range_sloppy(pc, s->imjd, s->fmjd, 1.05)) { return(-1); }

    /* Useful params */
    const double sec_per_sample = 1.0e-6/fabs(s->bw);
    const double days_per_sample = sec_per_sample/86400.0;
    const double fmjd0 = s->fmjd + days_per_sample * (double)s->overlap/2.0;
    const double days_per_fft = days_per_sample *
        (double)(s->fft_len - s->overlap);

    /* Temp mem */
    const size_t phase_mem = sizeof(double) * s->nfft_per_block;
    double *phase = (double *)malloc(phase_mem);
    double *step = (double *)malloc(phase_mem);

    int i;
    for (i=0; i<s->nfft_per_block; i++) {
        const double fmjd1 = fmjd0 + (double)i * days_per_fft;
        phase[i] = psr_phase(pc, s->imjd, fmjd1, NULL, NULL);
        phase[i] = fmod(phase[i], 1.0);
        if (phase[i]<0.0) phase[i] += 1.0;
        psr_phase(pc, s->imjd, fmjd1 + days_per_fft/2.0, &step[i], NULL);

        // Make units = phase bins
        phase[i] *= (double)s->nbins_fold;
        step[i] *= sec_per_sample * (double)s->nbins_fold;

        //printf("phase[%d]=%f step[%d]=%f inv_step[%d]=%f\n", 
        //        i, phase[i], i, step[i], i, 1.0/step[i]);
    }

    /* Copy over to GPU */
    cudaMemcpy(s->fold_phase, phase, phase_mem, cudaMemcpyHostToDevice);
    cudaMemcpy(s->fold_step, step, phase_mem, cudaMemcpyHostToDevice);

    free(phase);
    free(step);
    return(0);
}

/* Zero out all fold memory buffers in the GPU */
void zero_fold_buffers(struct dedispersion_setup *s) {

    const unsigned npol = 4;  // Assume always full-Stokes for now..
    const size_t bytes_per_profile = sizeof(float) * npol * s->nbins_fold;
    const size_t bytes_per_count = sizeof(unsigned) * s->nbins_fold;

    cudaMemset(s->foldtmp_gpu, 0, bytes_per_profile * 
            s->nfft_per_block);
    cudaMemset(s->foldtmp_c_gpu, 0, bytes_per_count * 
            s->nfft_per_block);
    //cudaMemset(s->foldbuf_gpu, 0, bytes_per_profile);
    //cudaMemset(s->foldbuf_c_gpu, 0, bytes_per_count);
}

/* Accumulate into a float4 stokes vector from two float2 pols */
inline __device__ void accStokes(float4 *stokes, float2 p0, float2 p1,
        unsigned *count) {
    stokes->x += p0.x*p0.x + p0.y*p0.y;
    stokes->y += p1.x*p1.x + p1.y*p1.y;
    stokes->z += p0.x*p1.x + p0.y*p1.y;
    stokes->w += p0.x*p1.y - p0.y*p1.x;
    (*count)++;
}

/* Fold each FFT chunk separately.
 * pol0, pol1 are input baseband data
 * Only works for 4 pol output.
 * Call with grid dims (nffts, nbins/BINS_PER_BLOCK)
 * All shared blocks need to fit into shared mem (16kB)
 */
#define BINS_PER_BLOCK 64
#define NTHREAD_FOLD BINS_PER_BLOCK
__global__ void fold_fft_blocks(const float2 *pol0, const float2 *pol1, 
        const double *phase, const double *step, 
        int fftlen, int overlap, int nbin,
        float4 *foldtmp, unsigned *foldtmp_c) {

    // Size params
    const int ifft = blockIdx.x;
    const int ibin = blockIdx.y;
    const int tid = threadIdx.x; // Thread index within the block
    const int nvalid = fftlen - overlap;

    // Pointers to start of valid data in global mem
    const float2 *ptr0 = pol0 + ifft*fftlen + overlap/2;
    const float2 *ptr1 = pol1 + ifft*fftlen + overlap/2;

    // Fold info
    const double bin0 = phase[ifft];
    const double bin_samp = step[ifft];   // bins/sample
    const double samp_bin = 1.0/bin_samp; // samples/bin
    const int bin_lo = ibin*BINS_PER_BLOCK + tid; // assumes 1 thread/bin
    const int nturn = ((double)nvalid*bin_samp)/(double)nbin + 2;

    // Fold results for this thread
    float4 folddata = make_float4(0,0,0,0);
    unsigned foldcount = 0;

    // Loop over number of pulse periods in data block
    for (int iturn=0; iturn<nturn; iturn++) {

        // Determine range of samples needed for this bin, turn
        int samp0 = samp_bin*((double)bin_lo-bin0+(double)iturn*nbin)+0.5;
        int samp1 = samp_bin*((double)bin_lo-bin0+(double)iturn*nbin+1)+0.5;

        // Range checks
        if (samp0<0) { samp0=0; }
        if (samp1<0) { samp1=0; }
        if (samp0>nvalid) { samp0=nvalid; }
        if (samp1>nvalid) { samp1=nvalid; }

        // Read in and add samples
        for (int isamp=samp0; isamp<samp1; isamp++) {
            float2 p0 = ptr0[isamp];
            float2 p1 = ptr1[isamp];
            folddata.x += p0.x*p0.x + p0.y*p0.y;
            folddata.y += p1.x*p1.x + p1.y*p1.y;
            folddata.z += p0.x*p1.x + p0.y*p1.y;
            folddata.w += p0.x*p1.y - p0.y*p1.x;
            foldcount++;
        }
    }

    // Copy results into global mem
    const unsigned prof_offset = ifft * nbin;
    foldtmp[prof_offset + bin_lo].x = folddata.x;
    foldtmp[prof_offset + bin_lo].y = folddata.y;
    foldtmp[prof_offset + bin_lo].z = folddata.z;
    foldtmp[prof_offset + bin_lo].w = folddata.w;
    foldtmp_c[prof_offset + bin_lo] = foldcount;
}

/* Combine folds.  Run with nbin threads */
__global__ void combine_folds(const float4 *foldtmp, const unsigned *foldtmp_c,
        float4 *foldbuf, unsigned *foldbuf_c, int nblock) {
    const int nt = blockDim.x * gridDim.x; // Should also equal nbin 
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    float4 ff;
    unsigned cc;
    ff = make_float4(0,0,0,0);
    cc=0;
    for (int i=tid; i<nblock*nt; i+=nt) {
        ff.x += foldtmp[i].x;
        ff.y += foldtmp[i].y;
        ff.z += foldtmp[i].z;
        ff.w += foldtmp[i].w;
        cc += foldtmp_c[i];
    }
    foldbuf[tid] = ff;
    foldbuf_c[tid] = cc;
}

/* Fold data in GPU.  Assumes dedisperse() has already been run,
 * ie there is valid data in GPU memory which should be folded.
 */
extern "C"
void fold(struct dedispersion_setup *s, int chan, struct foldbuf *fb) {

    /* Sizes */
    const unsigned npol = 4;  // Assume always full-Stokes for now..
    const size_t bytes_per_profile = sizeof(float) * npol * s->nbins_fold;
    const size_t bytes_per_count = sizeof(unsigned) * s->nbins_fold;

    /* Benchmarking */
#define NT 8
    cudaEvent_t t[NT];
    int it;
    for (it=0; it<NT; it++) cudaEventCreate(&t[it]);
    it=0;

    cudaEventRecord(t[it], 0); it++;
    cudaEventRecord(t[it], 0); it++;

    /* Zero out memory */
    zero_fold_buffers(s);
    cudaEventRecord(t[it], 0); it++;
    //cudaThreadSynchronize();
    //printf("fold cuda_err=\'%s\'\n", cudaGetErrorString(cudaGetLastError()));

    /* Fold FFT blocks */
    dim3 gd(s->nfft_per_block, s->nbins_fold/BINS_PER_BLOCK, 1);
    fold_fft_blocks<<<gd, NTHREAD_FOLD>>>(
            s->databuf0_gpu, s->databuf1_gpu, s->fold_phase, 
            s->fold_step, s->fft_len, s->overlap, s->nbins_fold,
            (float4 *)s->foldtmp_gpu, s->foldtmp_c_gpu);
    cudaEventRecord(t[it], 0); it++;
    //cudaThreadSynchronize();
    //printf("fold cuda_err=\'%s\'\n", cudaGetErrorString(cudaGetLastError()));

    /* Combine folds */
    combine_folds<<<s->nbins_fold/64,64>>>(
            (float4 *)s->foldtmp_gpu, s->foldtmp_c_gpu,
            (float4 *)s->foldbuf_gpu, s->foldbuf_c_gpu,
            s->nfft_per_block);
    cudaEventRecord(t[it], 0); it++;
    //cudaThreadSynchronize();
    //printf("fold cuda_err=\'%s\'\n", cudaGetErrorString(cudaGetLastError()));

    /* Transfer result to host */
    cudaMemcpy(fb->data + chan*fb->npol*fb->nbin,
            s->foldbuf_gpu, bytes_per_profile, 
            cudaMemcpyDeviceToHost);
    cudaMemcpy(fb->count + chan*fb->nbin,
            s->foldbuf_c_gpu, bytes_per_count, 
            cudaMemcpyDeviceToHost);
    cudaEventRecord(t[it], 0); it++;
    //cudaThreadSynchronize();
    //printf("fold cuda_err=\'%s\'\n", cudaGetErrorString(cudaGetLastError()));

    /* Final timer */
    cudaEventRecord(t[it], 0);
    cudaEventSynchronize(t[it]);
    cudaThreadSynchronize();
    //printf("fold cuda_err=\'%s\'\n", cudaGetErrorString(cudaGetLastError()));

    /* Add up timers */
    float ttmp;
    it=1;

    cudaEventElapsedTime(&ttmp, t[it], t[it+1]);
    s->time.fold_mem += ttmp;
    s->time.total2 += ttmp;
    it++;

    cudaEventElapsedTime(&ttmp, t[it], t[it+1]);
    s->time.fold_blocks += ttmp;
    s->time.total2 += ttmp;
    it++;

    cudaEventElapsedTime(&ttmp, t[it], t[it+1]);
    s->time.fold_combine += ttmp;
    s->time.total2 += ttmp;
    it++;

    cudaEventElapsedTime(&ttmp, t[it], t[it+1]);
    s->time.transfer_to_host += ttmp;
    s->time.total2 += ttmp;
    it++;

    cudaEventElapsedTime(&ttmp, t[0], t[it+1]);
    s->time.total += ttmp;

    /* Benchmark cleanup */
    for (it=0; it<NT; it++) cudaEventDestroy(t[it]);

}
