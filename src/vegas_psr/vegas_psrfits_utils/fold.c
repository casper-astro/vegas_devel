/* Simple fold routines */
#include <math.h>
#include <string.h>
#include <stdlib.h>
#include <pthread.h>

#ifdef FOLD_USE_INTRINSICS
#  include <xmmintrin.h>
#  define _MM_LOAD_PS  _mm_load_ps
#  define _MM_STORE_PS  _mm_store_ps
#endif

#include "fold.h"
#include "polyco.h"

void malloc_foldbuf(struct foldbuf *f) {
#ifdef FOLD_USE_INTRINSICS
    const int alignment = 64;
    if ((f->npol * f->nchan * sizeof(float)) % alignment) {
        fprintf(stderr, 
                "Error: foldbuf dimension are not appropriate for alignment:\n"
                "  npol=%d nchan=%d\n", f->npol, f->nchan);
        exit(1);
    }
    int rv = posix_memalign((void *)&f->data, alignment, 
            sizeof(float) * f->nbin * f->npol * f->nchan);
    if (rv) { 
        fprintf(stderr, "Error in posix_memalign");
        exit(1);
    }
#else
    f->data = (float *)malloc(sizeof(float) * f->nbin * f->npol * f->nchan);
#endif
    f->count = (unsigned *)malloc(sizeof(unsigned) * f->nbin);
}

void free_foldbuf(struct foldbuf *f) {
    if (f->data!=NULL) { free(f->data); f->data=NULL; }
    if (f->count!=NULL) { free(f->count); f->count=NULL; }
}

void clear_foldbuf(struct foldbuf *f) {
    memset(f->data, 0, sizeof(float) * f->nbin * f->npol * f->nchan);
    memset(f->count, 0, sizeof(unsigned) * f->nbin);
}

size_t foldbuf_data_size(const struct foldbuf *f) {
    if (f->data==NULL) return(0);
    return(sizeof(float) * f->nbin * f->npol * f->nchan);
}

size_t foldbuf_count_size(const struct foldbuf *f) {
    if (f->count==NULL) return(0);
    return(sizeof(unsigned) * f->nbin);
}

/* Combines unpack and accumulate */
void vector_accumulate_8bit(float *out, const char *in, int n) {
#ifdef FOLD_USE_INTRINSICS
    __m128 in_, out_, tmp_;
    float ftmp;
    int ii;
    for (ii = 0 ; ii < (n & -16) ; ii += 16) {
        __builtin_prefetch(out + 64, 1, 0);
        __builtin_prefetch(in  + 64, 0, 0);

        out_ = _MM_LOAD_PS(out);
        in_ = _mm_cvtpi8_ps(*((__m64 *)in));
        tmp_ = _mm_add_ps(out_, in_);
        _MM_STORE_PS(out, tmp_);
        in  += 4;
        out += 4;

        out_ = _MM_LOAD_PS(out);
        in_ = _mm_cvtpi8_ps(*((__m64 *)in));
        tmp_ = _mm_add_ps(out_, in_);
        _MM_STORE_PS(out, tmp_);
        in  += 4;
        out += 4;

        out_ = _MM_LOAD_PS(out);
        in_ = _mm_cvtpi8_ps(*((__m64 *)in));
        tmp_ = _mm_add_ps(out_, in_);
        _MM_STORE_PS(out, tmp_);
        in  += 4;
        out += 4;

        out_ = _MM_LOAD_PS(out);
        in_ = _mm_cvtpi8_ps(*((__m64 *)in));
        tmp_ = _mm_add_ps(out_, in_);
        _MM_STORE_PS(out, tmp_);
        in  += 4;
        out += 4;
    }
    for (; ii < (n & -4) ; ii += 4) {
        out_ = _MM_LOAD_PS(out);
        in_ = _mm_cvtpi8_ps(*((__m64 *)in));
        tmp_ = _mm_add_ps(out_, in_);
        _MM_STORE_PS(out, tmp_);
        in  += 4;
        out += 4;
    }
    for (; ii < n ; ii++) {  // Cast these without intrinsics
        ftmp = (float)(*in);
        out_ = _mm_load_ss(out);
        in_ = _mm_load_ss(&ftmp);
        tmp_ = _mm_add_ss(out_, in_);
        _mm_store_ss(out, tmp_);
        in  += 1;
        out += 1;
    }
    _mm_empty();
#else
    int i;
    for (i=0; i<n; i++) { out[i] += (float)in[i]; }
#endif
}

void vector_accumulate_8bit_unsigned(float *out, 
        const unsigned char *in, int n) {
#ifdef FOLD_USE_INTRINSICS
    __m128 in_, out_, tmp_;
    float ftmp;
    int ii;
    for (ii = 0 ; ii < (n & -16) ; ii += 16) {
        __builtin_prefetch(out + 64, 1, 0);
        __builtin_prefetch(in  + 64, 0, 0);

        out_ = _MM_LOAD_PS(out);
        in_ = _mm_cvtpu8_ps(*((__m64 *)in));
        tmp_ = _mm_add_ps(out_, in_);
        _MM_STORE_PS(out, tmp_);
        in  += 4;
        out += 4;

        out_ = _MM_LOAD_PS(out);
        in_ = _mm_cvtpu8_ps(*((__m64 *)in));
        tmp_ = _mm_add_ps(out_, in_);
        _MM_STORE_PS(out, tmp_);
        in  += 4;
        out += 4;

        out_ = _MM_LOAD_PS(out);
        in_ = _mm_cvtpu8_ps(*((__m64 *)in));
        tmp_ = _mm_add_ps(out_, in_);
        _MM_STORE_PS(out, tmp_);
        in  += 4;
        out += 4;

        out_ = _MM_LOAD_PS(out);
        in_ = _mm_cvtpu8_ps(*((__m64 *)in));
        tmp_ = _mm_add_ps(out_, in_);
        _MM_STORE_PS(out, tmp_);
        in  += 4;
        out += 4;
    }
    for (; ii < (n & -4) ; ii += 4) {
        out_ = _MM_LOAD_PS(out);
        in_ = _mm_cvtpu8_ps(*((__m64 *)in));
        tmp_ = _mm_add_ps(out_, in_);
        _MM_STORE_PS(out, tmp_);
        in  += 4;
        out += 4;
    }
    for (; ii < n ; ii++) {  // Cast these without intrinsics
        ftmp = (float)(*in);
        out_ = _mm_load_ss(out);
        in_ = _mm_load_ss(&ftmp);
        tmp_ = _mm_add_ss(out_, in_);
        _mm_store_ss(out, tmp_);
        in  += 1;
        out += 1;
    }
    _mm_empty();
#else
    int i;
    for (i=0; i<n; i++) { out[i] += (float)in[i]; }
#endif
}


void vector_accumulate(float *out, const float *in, int n) {
#ifdef FOLD_USE_INTRINSICS
    __m128 in_, out_, tmp_;
    int ii;
    for (ii = 0 ; ii < (n & -16) ; ii += 16) {
        __builtin_prefetch(out + 64, 1, 0);
        __builtin_prefetch(in  + 64, 0, 0);

        in_  = _MM_LOAD_PS(in);
        out_ = _MM_LOAD_PS(out);
        tmp_ = _mm_add_ps(out_, in_);
        _MM_STORE_PS(out, tmp_);
        in  += 4;
        out += 4;

        in_  = _MM_LOAD_PS(in);
        out_ = _MM_LOAD_PS(out);
        tmp_ = _mm_add_ps(out_, in_);
        _MM_STORE_PS(out, tmp_);
        in  += 4;
        out += 4;

        in_  = _MM_LOAD_PS(in);
        out_ = _MM_LOAD_PS(out);
        tmp_ = _mm_add_ps(out_, in_);
        _MM_STORE_PS(out, tmp_);
        in  += 4;
        out += 4;

        in_  = _MM_LOAD_PS(in);
        out_ = _MM_LOAD_PS(out);
        tmp_ = _mm_add_ps(out_, in_);
        _MM_STORE_PS(out, tmp_);
        in  += 4;
        out += 4;
    }
    for (; ii < (n & -4) ; ii += 4) {
        in_  = _MM_LOAD_PS(in);
        out_ = _MM_LOAD_PS(out);
        tmp_ = _mm_add_ps(out_, in_);
        _MM_STORE_PS(out, tmp_);
        in  += 4;
        out += 4;
    }
    for (; ii < n ; ii++) {
        in_  = _mm_load_ss(in);
        out_ = _mm_load_ss(out);
        tmp_ = _mm_add_ss(out_, in_);
        _mm_store_ss(out, tmp_);
        in  += 1;
        out += 1;
    }
    _mm_empty();
#else
    int i;
    for (i=0; i<n; i++) { out[i] += in[i]; }
#endif
}

int zero_check(const char *dat, int len) {
    int i, z=1;
    for (i=0; i<len; i++) { 
        if (dat[i]!='\0') { z=0; break; }
    }
    return(z);
}

void unpack_8bit(float *out, const char *in, int n) {
    int i;
    for (i=0; i<n; i++) { out[i] = (float)in[i]; }
}

void unpack_8bit_unsigned(float *out, const unsigned char *in, int n) {
    int i;
    for (i=0; i<n; i++) { out[i] = (float)in[i]; }
}

void *fold_8bit_power_thread(void *_args) {
    struct fold_args *args = (struct fold_args *)_args;
    int rv = fold_8bit_power(args->pc, args->imjd, args->fmjd, args->data,
            args->nsamp, args->tsamp, args->raw_signed, args->fb);
    pthread_exit(&rv);
}

int fold_8bit_power(const struct polyco *pc, int imjd, double fmjd, 
        const char *data, int nsamp, double tsamp, int raw_signed,
        struct foldbuf *f) {

    /* Find midtime */
    double fmjd_mid = fmjd + nsamp*tsamp/2.0/86400.0;

    /* Check polyco set, allow 5% expansion of range */
    if (pc_out_of_range_sloppy(pc, imjd, fmjd,1.05)) { return(-1); }

    /* Calc phase, phase step */
    /* NOTE: Starting sample phase is computed for the middle
     * of the first sample, assuming input fmjd refers to 
     * the rising edge of the first sample given
     */
    double dphase=0.0;
    double phase = psr_phase(pc, imjd, fmjd + tsamp/2.0/86400.0, NULL, NULL);
    phase = fmod(phase, 1.0);
    if (phase<0.0) { phase += 1.0; }
    psr_phase(pc, imjd, fmjd_mid, &dphase, NULL);
    dphase *= tsamp;

    /* Fold em */
    int i, ibin;
    float *fptr;
    for (i=0; i<nsamp; i++) {
        ibin = (int)(phase * (double)f->nbin);
        if (ibin<0) { ibin+=f->nbin; }
        if (ibin>=f->nbin) { ibin-=f->nbin; }
        fptr = &f->data[ibin*f->nchan*f->npol];
        if (zero_check(&data[i*f->nchan*f->npol],f->nchan*f->npol)==0) { 
            if (raw_signed==1)
                vector_accumulate_8bit(fptr, 
                        &data[i*f->nchan*f->npol],
                        f->nchan*f->npol);
            else if (raw_signed==2) {
                // First 2 polns are unsigned, last 2 are signed
                vector_accumulate_8bit_unsigned(fptr, 
                        &data[i*f->nchan*f->npol],
                        f->nchan*2);
                vector_accumulate_8bit(fptr + 2*f->nchan, 
                        &data[i*f->nchan*f->npol + 2*f->nchan],
                        f->nchan*2);
            } else if (raw_signed==3) {
                // First 1 pol is unsigned, last 3 are signed
                vector_accumulate_8bit_unsigned(fptr, 
                        &data[i*f->nchan*f->npol],
                        f->nchan);
                vector_accumulate_8bit(fptr + f->nchan, 
                        &data[i*f->nchan*f->npol + f->nchan],
                        f->nchan*3);
            } else 
                // All unsigned
                vector_accumulate_8bit_unsigned(fptr, 
                        (unsigned char *)&data[i*f->nchan*f->npol],
                        f->nchan*f->npol);
            f->count[ibin]++;
        }
        phase += dphase;
        if (phase>1.0) { phase -= 1.0; }
    }

    return(0);
}

int accumulate_folds(struct foldbuf *ftot, const struct foldbuf *f) {
    if (ftot->nbin!=f->nbin || ftot->nchan!=f->nchan || ftot->npol!=f->npol) {
        return(-1);
    }
    int i;
    for (i=0; i<f->nbin; i++) { ftot->count[i] += f->count[i]; }
    vector_accumulate(ftot->data, f->data, f->nbin * f->nchan * f->npol);
    return(0);
}

/* normalize and transpose to psrfits order */
int normalize_transpose_folds(float *out, const struct foldbuf *f) {
    int ibin, ii;
    for (ibin=0; ibin<f->nbin; ibin++) {
        if (f->count[ibin]==0) {
            for (ii=0; ii<f->nchan*f->npol; ii++) 
                out[ibin + ii*f->nbin] = 0.0;
        } else {
            for (ii=0; ii<f->nchan*f->npol; ii++) 
                out[ibin + ii*f->nbin] =
                    f->data[ii + ibin*f->nchan*f->npol] / (float)f->count[ibin];
        }
    }
    return(0);
}

/* Apply a per-channel/poln scale and offset */
/* Note, this only works on pre-normalized fold results! */
int scale_offset_folds(struct foldbuf *f, 
        const float *scale, const float *offset) {
    int ibin, ii;
    float tmp;
    for (ibin=0; ibin<f->nbin; ibin++) {
        for (ii=0; ii<f->nchan*f->npol; ii++) {
            tmp = f->data[ii + ibin*f->nchan*f->npol]; 
            f->data[ii + ibin*f->nchan*f->npol] = tmp*scale[ii] + 
                (float)f->count[ibin]*offset[ii]; 
        }
    }
    return(0);
}
