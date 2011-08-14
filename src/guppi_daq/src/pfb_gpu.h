#ifndef _PFB_GPU_H
#define _PFB_GPU_H

#include "guppi_databuf.h"

#define FALSE               0
#define TRUE                1

#define VEGAS_NUM_TAPS      8

#define FILE_COEFF_PREFIX   "coeff"
#define FILE_COEFF_DATATYPE "signedchar"
#define FILE_COEFF_SUFFIX   ".dat"

#define FFTPLAN_RANK        1
#define FFTPLAN_ISTRIDE     (2 * g_num_subbands)
#define FFTPLAN_OSTRIDE     (2 * g_num_subbands)
#define FFTPLAN_IDIST       1
#define FFTPLAN_ODIST       1
#define FFTPLAN_BATCH       (2 * g_num_subbands)

#if defined __cplusplus
extern "C"
#endif
int init_gpu(size_t input_block_sz, size_t output_block_sz, size_t index_sz, int num_subbands, int num_chans);

#if defined __cplusplus
extern "C"
#endif
void do_pfb(struct guppi_databuf *db_in,
            int curblock_in,
            struct guppi_databuf *db_out,
            int first,
            struct guppi_status st,
            int acc_len);

int do_fft();

int accumulate();

void zero_accumulator();

int get_accumulated_spectrum_from_device(char *out);

int is_valid(int heap_start, int num_heaps);

int is_blanked(int heap_start, int num_heaps);

/* Free up any allocated memory */
void cleanup_gpu();

#endif

