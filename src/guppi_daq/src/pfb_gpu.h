#ifndef _PFB_GPU_H
#define _PFB_GPU_H

#include "guppi_databuf.h"

#define FALSE               0
#define TRUE                1

#define VEGAS_NUM_TAPS      8

#define FFTPLAN_RANK        1
#define FFTPLAN_ISTRIDE     (2 * g_iNumSubBands)
#define FFTPLAN_OSTRIDE     (2 * g_iNumSubBands)
#define FFTPLAN_IDIST       1
#define FFTPLAN_ODIST       1
#define FFTPLAN_BATCH       (2 * g_iNumSubBands)

#if defined __cplusplus
extern "C"
#endif
void init_gpu(size_t input_block_sz, size_t output_block_sz, size_t index_sz, int num_subbands, int num_chans);

#if defined __cplusplus
extern "C"
#endif
void do_pfb(char *in,
            char *out,
            struct databuf_index* index_in,
            struct databuf_index* index_out,
            int first);

int do_fft();

int accumulate();

void zero_accumulator();

int get_accumulated_spectrum_from_device(char *out);

/* Free up any allocated memory */
void cleanup_gpu();

#endif

