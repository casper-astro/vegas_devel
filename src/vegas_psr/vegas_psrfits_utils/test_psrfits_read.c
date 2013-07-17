#include <stdio.h>
#include <stdlib.h>
#include "psrfits.h"

int main(int argc, char *argv[]) {
    struct psrfits pf;
    sprintf(pf.basefilename, 
            "/data2/demorest/parspec/parspec_test_B0329+54_0009");
    pf.filenum=1;
    int rv = psrfits_open(&pf);
    pf.sub.dat_freqs = (float *)malloc(sizeof(float) * pf.hdr.nchan);
    pf.sub.dat_weights = (float *)malloc(sizeof(float) * pf.hdr.nchan);
    pf.sub.dat_offsets = (float *)malloc(sizeof(float) 
            * pf.hdr.nchan * pf.hdr.npol);
    pf.sub.dat_scales  = (float *)malloc(sizeof(float) 
            * pf.hdr.nchan * pf.hdr.npol);
    pf.sub.data  = (unsigned char *)malloc(pf.sub.bytes_per_subint);
    while ((rv=psrfits_read_subint(&pf))==0) { 
        printf("Read subint (file %d, row %d/%d)\n", 
                pf.filenum, pf.rownum-1, pf.rows_per_file);
    }
    if (rv) { fits_report_error(stderr, rv); }
    exit(0);
}
