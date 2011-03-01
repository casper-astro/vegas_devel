#ifndef _FOLD_H
#define _FOLD_H
#include "polyco.h"

/* Defines ordering of fold buf data.
 * First dim mentioned runs fastest in memory, etc..
 */
enum fold_order {
    chan_pol_bin, // For CPU/SSE folding
    pol_bin_chan, // For GPU folding
    bin_chan_pol  // For PSRFITS output
};

struct foldbuf {
    int nbin;
    int nchan;
    int npol;
    enum fold_order order;
    float *data;
    unsigned *count;
};

void malloc_foldbuf(struct foldbuf *f);

void free_foldbuf(struct foldbuf *f);

void clear_foldbuf(struct foldbuf *f);

size_t foldbuf_data_size(const struct foldbuf *f);
size_t foldbuf_count_size(const struct foldbuf *f);

void scale_counts(struct foldbuf *f, float fac);

int normalize_transpose_folds(float *out, const struct foldbuf *f);

struct fold_args {
    struct polyco *pc;
    int imjd;
    double fmjd;
    char *data;
    int nsamp;
    double tsamp;
    int raw_signed;
    struct foldbuf *fb;
};

void *fold_8bit_power_thread(void *_args);

int fold_8bit_power(const struct polyco *pc, int imjd, double fmjd, 
        const char *data, int nsamp, double tsamp, int raw_signed,
        struct foldbuf *f);

int accumulate_folds(struct foldbuf *ftot, const struct foldbuf *f);

#endif
