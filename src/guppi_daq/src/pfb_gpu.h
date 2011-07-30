#ifndef _PFB_GPU_H
#define _PFB_GPU_H

#include "guppi_databuf.h"

/* Initialize all necessary memory, etc for doing PFB 
 * at the given params.
 */
void init_pfb(size_t input_block_sz, size_t output_block_sz, size_t index_sz, int num_chans);

/* Actually do the PFB by calling CUDA kernels */
void do_pfb(const char *in, char *out, struct databuf_index* index_in,
            struct databuf_index* index_out);

/* Free up any allocated memory */
void destroy_pfb();

#endif
