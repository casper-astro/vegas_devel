#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "psrfits.h"
#include "slalib.h"

#ifndef DEGTORAD
#define DEGTORAD 0.017453292519943295769236907684886127134428718885417
#endif
#ifndef RADTODEG
#define RADTODEG 57.29577951308232087679815481410517033240547246656
#endif

void dec2hms(char *out, double in, int sflag) {
    int sign = 1;
    char *ptr = out;
    int h, m;
    double s;
    if (in<0.0) { sign = -1; in = fabs(in); }
    h = (int)in; in -= (double)h; in *= 60.0;
    m = (int)in; in -= (double)m; in *= 60.0;
    s = in;
    if (sign==1 && sflag) { *ptr='+'; ptr++; }
    else if (sign==-1) { *ptr='-'; ptr++; }
    sprintf(ptr, "%2.2d:%2.2d:%07.4f", h, m, s);
}

int main(int argc, char *argv[]) {
    int ii;
    double dtmp;
    struct psrfits pf;
    
    // Only set the basefilename and not "filename"
    // Also, fptr will be set by psrfits_create_searchmode()
    
    strcpy(pf.basefilename, "test_psrfits");
    pf.filenum = 0;           // This is the crucial one to set to initialize things
    pf.rows_per_file = 200;  // Need to set this based on PSRFITS_MAXFILELEN

    // Now set values for our hdrinfo structure
    pf.hdr.scanlen = 5; // in sec
    strcpy(pf.hdr.observer, "John Doe");
    strcpy(pf.hdr.source, "Cool PSR A");
    strcpy(pf.hdr.frontend, "L-band");
    strcpy(pf.hdr.project_id, "GBT09A-001");
    strcpy(pf.hdr.date_obs, "2010-01-01T05:15:30.000");
    strcpy(pf.hdr.poln_type, "LIN");
    strcpy(pf.hdr.track_mode, "TRACK");
    strcpy(pf.hdr.cal_mode, "OFF");
    strcpy(pf.hdr.feed_mode, "FA");
    pf.hdr.dt = 0.000050;
    pf.hdr.fctr = 1400.0;
    pf.hdr.BW = 800.0;
    pf.hdr.ra2000 = 302.0876876;
    dec2hms(pf.hdr.ra_str, pf.hdr.ra2000/15.0, 0);
    pf.hdr.dec2000 = -3.456987698;
    dec2hms(pf.hdr.dec_str, pf.hdr.dec2000, 1);
    pf.hdr.azimuth = 123.123;
    pf.hdr.zenith_ang = 23.0;
    pf.hdr.beam_FWHM = 0.25;
    pf.hdr.start_lst = 10000.0;
    pf.hdr.start_sec = 25000.82736876;
    pf.hdr.start_day = 55000;
    pf.hdr.scan_number = 3;
    pf.hdr.rcvr_polns = 2;
    pf.hdr.summed_polns = 1;
    pf.hdr.offset_subint = 0;
    pf.hdr.nchan = 1024;
    pf.hdr.orig_nchan = pf.hdr.nchan;
    pf.hdr.orig_df = pf.hdr.df = pf.hdr.BW / pf.hdr.nchan;
    pf.hdr.nbits = 8;
    pf.hdr.npol = 1;
    pf.hdr.nsblk = 10000;
    pf.hdr.MJD_epoch = 55555.123123123123123123L;  // Note the "L" for long double

    // Now set values for our subint structure
    pf.sub.tsubint = pf.hdr.nsblk * pf.hdr.dt;
    pf.sub.offs = (pf.tot_rows + 0.5) * pf.sub.tsubint;
    pf.sub.lst = pf.hdr.start_lst;
    pf.sub.ra = pf.hdr.ra2000;
    pf.sub.dec = pf.hdr.dec2000;
    slaEqgal(pf.hdr.ra2000*DEGTORAD, pf.hdr.dec2000*DEGTORAD, 
             &pf.sub.glon, &pf.sub.glat);
    pf.sub.glon *= RADTODEG;
    pf.sub.glat *= RADTODEG;
    pf.sub.feed_ang = 0.0;
    pf.sub.pos_ang = 0.0;
    pf.sub.par_ang = 0.0;
    pf.sub.tel_az = pf.hdr.azimuth;
    pf.sub.tel_zen = pf.hdr.zenith_ang;
    pf.sub.bytes_per_subint = (pf.hdr.nbits * pf.hdr.nchan * 
                               pf.hdr.npol * pf.hdr.nsblk) / 8;
    pf.sub.FITS_typecode = TBYTE;  // 11 = byte

    // Create and initialize the subint arrays
    pf.sub.dat_freqs = (float *)malloc(sizeof(float) * pf.hdr.nchan);
    pf.sub.dat_weights = (float *)malloc(sizeof(float) * pf.hdr.nchan);
    dtmp = pf.hdr.fctr - 0.5 * pf.hdr.BW + 0.5 * pf.hdr.df;
    for (ii = 0 ; ii < pf.hdr.nchan ; ii++) {
        pf.sub.dat_freqs[ii] = dtmp + ii * pf.hdr.df;
        pf.sub.dat_weights[ii] = 1.0;
    }
    pf.sub.dat_offsets = (float *)malloc(sizeof(float) * pf.hdr.nchan * pf.hdr.npol);
    pf.sub.dat_scales = (float *)malloc(sizeof(float) * pf.hdr.nchan * pf.hdr.npol);
    for (ii = 0 ; ii < pf.hdr.nchan * pf.hdr.npol ; ii++) {
        pf.sub.dat_offsets[ii] = 0.0;
        pf.sub.dat_scales[ii] = 1.0;
    }
 

    // This is what you would update for each time sample (likely just
    // adjusting the pointer to point to your data)
    pf.sub.data = (unsigned char *)malloc(pf.sub.bytes_per_subint);
    for (ii = 0 ; ii < pf.sub.bytes_per_subint ; ii++) {
        pf.sub.data[ii] = ii % 256;
    }

    // Here is the real data-writing loop
    do {
        // Update the pf.sub entries here for each subint
        // as well as the pf.sub.data pointer
        psrfits_write_subint(&pf);
    } while (pf.T < pf.hdr.scanlen && !pf.status);

    // Close the last file and cleanup
    fits_close_file(pf.fptr, &(pf.status));
    free(pf.sub.dat_freqs);
    free(pf.sub.dat_weights);
    free(pf.sub.dat_offsets);
    free(pf.sub.dat_scales);
    free(pf.sub.data);

    printf("Done.  Wrote %d subints (%f sec) in %d files.  status = %d\n", 
           pf.tot_rows, pf.T, pf.filenum, pf.status);

    exit(0);
}
