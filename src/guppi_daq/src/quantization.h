/* quantization.h */
#ifndef _QUANTIZATION_H
#define _QUANTIZATION_H
#include "psrfits.h"
inline int quantize_2bit(struct psrfits *pf, double * mean, double * std);
int compute_stat(struct psrfits *pf, double *mean, double *std);
#endif
