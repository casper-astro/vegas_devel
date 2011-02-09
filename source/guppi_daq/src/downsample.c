#include <math.h>
#include <string.h>
#include "psrfits.h"

// TODO:  for these to work with OpenMP, we probably need
//        separate input and output arrays and then a copy.
//        Otherwise, the threads will step on each other.

void get_stokes_I(struct psrfits *pf)
/* Move the Stokes I in place so that it is consecutive in the array */
{
    int ii, inbytes, outbytes;
    struct hdrinfo *hdr = &(pf->hdr);
    unsigned char *data = pf->sub.data;

    outbytes = hdr->nbits * hdr->nchan / 8;
    inbytes = outbytes * 4;  // 4 Stokes params
    // Start from 1 since we don't need to move the 1st spectra
    for (ii = 1 ; ii < hdr->nsblk ; ii++)
        memcpy(data + ii * outbytes, data + ii * inbytes, outbytes);
}

void downsample_freq(struct psrfits *pf)
/* Average adjacent frequency channels together in place    */
/* Note:  this only works properly for 8-bit data currently */
{
    int ii, jj, itmp;
    struct hdrinfo *hdr = &(pf->hdr);
    char *indata = (char *)pf->sub.data;
    char *outdata = (char *)pf->sub.data;
    const int dsfact = hdr->ds_freq_fact;
    const int offset = dsfact >> 1;
    const int shift = ffs(dsfact) - 1; // dsfact's power of 2

    // Treat the polns as being parts of the same spectrum
    int out_npol = hdr->npol;
    if (hdr->onlyI) out_npol = 1;
    const int out_nchan = hdr->nchan * out_npol / hdr->ds_freq_fact;
    
    // Iterate over the times and output chans
    for (ii = 0 ; ii < hdr->nsblk * out_nchan ; ii++) {
        // and over adjacent input chans for each time
        for (jj = 0, itmp = offset ; jj < dsfact ; jj++)
            itmp += *indata++;
        // The following adds 1/2 of dsfact to the total (which allows
        // for rounding-type behavior) and then divides by dsfact (via
        // a bit shift of the appropriate number of bits, since dsfact
        // should be a power-of-two).
        *outdata++ = itmp >> shift;
    }
}

void downsample_time(struct psrfits *pf)
/* Average adjacent time samples together in place */
/* This should be called _after_ downsample_freq() */
/* Note:  this only works properly for 8-bit data currently */
{
    int ii, jj, kk, itmp, chanoff1, chanoff2;
    struct hdrinfo *hdr = &(pf->hdr);
    char *data = (char *)pf->sub.data;
    char *indata, *outdata;
    const int dsfact = hdr->ds_time_fact;
    const int offset = dsfact >> 1;
    const int shift = ffs(dsfact) - 1; // dsfact's power of 2

    // Treat the polns as being parts of the same spectrum
    int out_npol = hdr->npol;
    if (hdr->onlyI) out_npol = 1;
    const int out_nchan = hdr->nchan * out_npol / hdr->ds_freq_fact;
    const int out_nsblk = hdr->nsblk / dsfact;
    
    // Iterate over the output times
    for (ii = 0 ; ii < out_nsblk ; ii++) {
        chanoff1 = ii * out_nchan;
        chanoff2 = chanoff1 * dsfact;
        outdata = data + chanoff1;
        // and over each channel
        for (jj = 0 ; jj < out_nchan ; jj++) {
            indata = data + chanoff2 + jj;
            // to add the adjacent times
            for (kk = 0, itmp = offset ; kk < dsfact ; kk++) {
                itmp += *indata;
                indata += out_nchan;
            }
            *outdata++ = itmp >> shift;
        }
    }
}

void guppi_update_ds_params(struct psrfits *pf)
/* Update the various output data arrays / values so that */
/* they are correct for the downsampled data.             */
{
    struct hdrinfo *hdr = &(pf->hdr);
    struct subint  *sub = &(pf->sub);

    int out_npol = hdr->npol;
    if (hdr->onlyI) out_npol = 1;
    int out_nchan = hdr->nchan / hdr->ds_freq_fact;
 
    if (hdr->ds_freq_fact > 1) {
        int ii;
        double dtmp;

        /* Note:  we don't need to malloc the subint arrays since */
        /*        their original values are longer by default.    */

        // The following correctly accounts for the middle-of-bin FFT offset
        dtmp = hdr->fctr - 0.5 * hdr->BW;
        dtmp += 0.5 * (hdr->ds_freq_fact - 1.0) * hdr->df;
        for (ii = 0 ; ii < out_nchan ; ii++)
            sub->dat_freqs[ii] = dtmp + ii * (hdr->df * hdr->ds_freq_fact);

        for (ii = 1 ; ii < out_npol ; ii++) {
            memcpy(sub->dat_offsets+ii*out_nchan,
                   sub->dat_offsets+ii*hdr->nchan,
                   sizeof(float)*out_nchan);
            memcpy(sub->dat_scales+ii*out_nchan,
                   sub->dat_scales+ii*hdr->nchan,
                   sizeof(float)*out_nchan);
        }
    }
}
