/* cyclic_gpu.cu
 * Cyclic spectra in CUDA/GPU
 * P. Demorest, 2010/02
 */
#include <math.h>
#include "dedisperse_gpu.h"
#include "polyco.h"

extern "C"
void init_cyclic(struct dedispersion_setup *s) {
    // TODO: alloc any extra memory, etc.
}

extern "C"
int compute_cyclic_params(struct dedispersion_setup *s, 
        const struct polyco *pc) {
    // TODO compute psr freq, phase for current data
    // maybe just reuse compute_fold_params()...
}

/*
 * This func expects dedispersed data which has not
 * been IFFT'd back to time domain.
 * Try using one thread per output channel.
 * Block dims (nchan/nthread, nfft)
 */
__global__ void cyclic(const float2 *pol0, const float2 *pol1,
        const double *phase, const double *freq,
        int fftlen, int overlap, int nharm, int nchan,
        float4 *output) {

    // Sizes
    const int ifft = blockIdx.y;
    const int ichan = threadIdx.x + blockDim.x*blockIdx.x;

    // Spin info at midpoint of fft block
    const double phase_mid = phase[ifft];
    const double freq_mid = freq[ifft];

    // Output point
    float4 result = make_float4(0,0,0,0);
    unsigned result_count = 0;

}
