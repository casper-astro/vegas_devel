/* dedisperse_gpu.h */
#ifndef _DEDISPERSE_GPU_H
#define _DEDISPERSE_GPU_H
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda.h>
#include <cufft.h>
#include "guppi_params.h"

/* Time different operations for benchmarks */
struct dedispersion_times {
    double transfer_to_gpu;
    double overlap;
    double bit_to_float;
    double fft;
    double xmult;
    double detect;
    double transfer_to_host;
    double fold_mem;
    double fold_blocks;
    double fold_combine;
    double downsample;
    double total;
    double total2;
    unsigned long long nsamp_tot;
};

/* Describes dedispersion params */
#define MAX_CHAN 8192
struct dedispersion_setup {

    // Basic params
    double rf;        // Nominal center freq of band, MHz
    double bw;        // Channel bandwidth (including sign), MHz
    double dm;        // Dispersion measure, pc/cm^3
    double earth_z4;  // Earth-motion doppler correction
    unsigned nchan;   // Number of different channels to (potentially) dedisp
    unsigned npts_per_block;  // Number of samples sent at a time
    unsigned fft_len;         // FFT length to use
    unsigned overlap;         // Overlap between FFTs to use
    unsigned nfft_per_block;  // FFTs per block (per poln)
    double freq[MAX_CHAN];    // Freq values for each chan (MHz)

    // Info for folding
    int imjd;             // MJD of current first input data point
    double fmjd;          // "
    unsigned nbins_fold;  // # of bins per pulse period to use
    double *fold_phase;   // Pulse phase at start of each FFT in block (bins)
    double *fold_step;    // Step size (bins/sample) for each FFT

    // Info for downsampling
    int dsfac;            // Downsample factor
    int npol;             // Number of polarizations to save (4 or 1)

    // Memory blocks, etc on the host and/or GPU
    unsigned char *tbuf_host;     // host memory for data transfer
    unsigned char *tbuf_gpu;      // gpu memory for data transfer
    unsigned char *overlap_gpu;   // overlapped raw data
    float2 *databuf0_gpu;         // floating-point data on gpu, pol 0
    float2 *databuf1_gpu;         // floating-point data on gpu, pol 1
    float2 *chirp_gpu[MAX_CHAN];  // chirp func on gpu
    float *result_host;           // Post-dedispersion results

    // Memory for folding
    float *foldtmp_gpu;           // Working space for folding
    unsigned *foldtmp_c_gpu;      // Working space for folding (counts)
    float *foldbuf_gpu;           // Final folded data
    unsigned *foldbuf_c_gpu;      // Final folded data (counts)

    // Memory for downsampling
    char *dsbuf_gpu;              // 8-bit downsampled data

    // GPU control stuff
    cufftHandle plan;           // CUFFT plan
    int fold_thread_per_block;  // Thread info for folding.

    // Benchmark
    struct dedispersion_times time;
    struct guppi_params *gp;

};

#ifdef __cplusplus
extern "C" {
#endif
void init_chirp(struct dedispersion_setup *s);
void init_dedispersion(struct dedispersion_setup *s);
void dedisperse(struct dedispersion_setup *s, int ichan,
        const unsigned char *in, float *out);
void free_dedispersion(struct dedispersion_setup *s);
void print_timing_report(struct dedispersion_setup *s);
#ifdef __cplusplus
}
#endif

#endif
