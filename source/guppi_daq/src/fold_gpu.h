#ifndef _FOLD_GPU_H
#define _FOLD_GPU_H
#include "dedisperse_gpu.h"
#include "fold.h"
#ifdef __cplusplus
extern "C" {
#endif
void init_fold(struct dedispersion_setup *s);
int compute_fold_params(struct dedispersion_setup *s, const struct polyco *pc);
void fold(struct dedispersion_setup *s, int chan, struct foldbuf *fb_out);
#ifdef __cplusplus
}
#endif
#endif
