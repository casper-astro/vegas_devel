/* write_psrfits.c */
#define _ISOC99_SOURCE  // For long double strtold
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>
#include "sdfits.h"

#define DEBUGOUT 0


int sdfits_create(struct sdfits *sf) {
    int itmp, *status;
    char ctmp[40];
    struct hdrinfo *hdr;

    hdr = &(sf->hdr);        // dereference the ptr to the header struct
    status = &(sf->status);  // dereference the ptr to the CFITSIO status

    // Initialize the key variables if needed
    if (sf->new_file == 1) {  // first time writing to the file
        sf->status = 0;
        sf->tot_rows = 0;
        sf->N = 0L;
        sf->T = 0.0;
        sf->mode = 'w';

        // Create the output directory if needed
        char datadir[1024];
        strncpy(datadir, sf->basefilename, 1023);
        char *last_slash = strrchr(datadir, '/');
        if (last_slash!=NULL && last_slash!=datadir) {
            *last_slash = '\0';
            printf("Using directory '%s' for output.\n", datadir);
            char cmd[1024];
            sprintf(cmd, "mkdir -m 1777 -p %s", datadir);
            system(cmd);
        }
        sf->new_file = 0;
    }
    sf->filenum++;
    sf->rownum = 1;

    sprintf(sf->filename, "%s_%04d.fits", sf->basefilename, sf->filenum);

    // Create basic FITS file from our template
    char *vegas_dir = getenv("VEGAS_DIR");
    char template_file[1024];
    if (vegas_dir==NULL) {
        fprintf(stderr, 
                "Error: VEGAS_DIR environment variable not set, exiting.\n");
        exit(1);
    }
    printf("Opening file '%s'\n", sf->filename);
    sprintf(template_file, "%s/%s", vegas_dir, SDFITS_TEMPLATE);
    fits_create_template(&(sf->fptr), sf->filename, template_file, status);

    // Check to see if file was successfully created
    if (*status) {
        fprintf(stderr, "Error creating sdfits file from template.\n");
        fits_report_error(stderr, *status);
        exit(1);
    }

    // Go to the primary HDU
    fits_movabs_hdu(sf->fptr, 1, NULL, status);

    // Update the keywords that need it
    fits_get_system_time(ctmp, &itmp, status);      // date the file was written
    fits_update_key(sf->fptr, TSTRING, "DATE", ctmp, NULL, status);

    // Go to the SINGLE DISH HDU
    fits_movnam_hdu(sf->fptr, BINARY_TBL, "SINGLE DISH", 0, status);

    // Update the keywords that need it
    fits_update_key(sf->fptr, TSTRING, "TELESCOP", hdr->telescope,NULL, status);
    fits_update_key(sf->fptr, TDOUBLE, "BANDWID", &(hdr->bandwidth), NULL, status);
    fits_update_key(sf->fptr, TSTRING, "DATE-OBS", hdr->date_obs, NULL, status);
    fits_update_key(sf->fptr, TDOUBLE, "TSYS", &(hdr->tsys), NULL, status);

    fits_update_key(sf->fptr, TSTRING, "PROJID", hdr->projid, NULL, status);
    fits_update_key(sf->fptr, TSTRING, "FRONTEND", hdr->frontend, NULL, status);
    fits_update_key(sf->fptr, TDOUBLE, "OBSFREQ", &(hdr->obsfreq), NULL, status);
    fits_update_key(sf->fptr, TDOUBLE, "SCAN", &(hdr->scan), NULL, status);

    fits_update_key(sf->fptr, TSTRING, "INSTRUME", hdr->instrument, NULL, status);
    fits_update_key(sf->fptr, TSTRING, "CAL_MODE", hdr->cal_mode, NULL, status);
    if (strcmp("OFF", hdr->cal_mode) != 0)
    {
        fits_update_key(sf->fptr, TDOUBLE, "CAL_FREQ", &(hdr->cal_freq), NULL, status);
        fits_update_key(sf->fptr, TDOUBLE, "CAL_DCYC", &(hdr->cal_dcyc), NULL, status);
        fits_update_key(sf->fptr, TDOUBLE, "CAL_PHS", &(hdr->cal_phs), NULL, status);
    }
    fits_update_key(sf->fptr, TINT,    "NPOL", &(hdr->npol), NULL, status);
    fits_update_key(sf->fptr, TINT,    "NCHAN", &(hdr->nchan), NULL, status);
    fits_update_key(sf->fptr, TDOUBLE, "CHAN_BW", &(hdr->chan_bw), NULL, status);
    fits_update_key(sf->fptr, TINT,    "NSUBBAND", &(hdr->nsubband), NULL, status);
    fits_update_key(sf->fptr, TDOUBLE, "EFSAMPFR", &(hdr->efsampfr), NULL, status);
    fits_update_key(sf->fptr, TDOUBLE, "FPGACLK", &(hdr->fpgaclk), NULL, status);
    fits_update_key(sf->fptr, TDOUBLE, "HWEXPOSR", &(hdr->hwexposr), NULL, status);
    fits_update_key(sf->fptr, TDOUBLE, "FILTNEP", &(hdr->filtnep), NULL, status);
    fits_update_key(sf->fptr, TDOUBLE, "STTMJD", &(hdr->sttmjd), NULL, status);

    // Update the column sizes for the colums containing arrays
    itmp = hdr->nsubband * hdr->nchan * 4;                          //num elements, not bytes
    fits_modify_vector_len(sf->fptr, 20, itmp, status);             // DATA
    fits_modify_vector_len(sf->fptr, 14, hdr->nsubband, status);    // SUBFREQ

    // Update the TDIM field for the data column
    sprintf(ctmp, "(%d,%d,4,1,1)", hdr->nchan, hdr->nsubband);
    fits_update_key(sf->fptr, TSTRING, "TDIM20", ctmp, NULL, status);

    fits_flush_file(sf->fptr, status);
   
    return *status;
}


int sdfits_write_subint(struct sdfits *sf) {
    int row, *status;
    int nchan, nivals, nsubband;
    struct hdrinfo *hdr;
    struct sdfits_data_columns *dcols;
    char* temp_str;
    double temp_dbl;

    hdr = &(sf->hdr);               // dereference the ptr to the header struct
    dcols = &(sf->data_columns);    // dereference the ptr to the subint struct
    status = &(sf->status);         // dereference the ptr to the CFITSIO status
    nchan = hdr->nchan; 
    nsubband = hdr->nsubband;
    nivals = nchan * nsubband * 4;  // 4 stokes parameters

    // Create the initial file or change to a new one if needed.
    if (sf->new_file || (sf->multifile==1 && sf->rownum > sf->rows_per_file))
    {
        if (!sf->new_file) {
            printf("Closing file '%s'\n", sf->filename);
            fits_close_file(sf->fptr, status);
        }
        sdfits_create(sf);
    }
    row = sf->rownum;
    temp_str = dcols->object;
    temp_dbl = 0.0;
    dcols->centre_freq_idx++;

    fits_write_col(sf->fptr, TDOUBLE, 1,  row, 1, 1, &(dcols->time), status);
    fits_write_col(sf->fptr, TINT,    2,  row, 1, 1, &(dcols->time_counter), status);
    fits_write_col(sf->fptr, TINT,    3,  row, 1, 1, &(dcols->integ_num), status);
    fits_write_col(sf->fptr, TFLOAT,  4,  row, 1, 1, &(dcols->exposure), status);
    fits_write_col(sf->fptr, TSTRING, 5,  row, 1, 1, &temp_str, status);
    fits_write_col(sf->fptr, TFLOAT,  6,  row, 1, 1, &(dcols->azimuth), status);
    fits_write_col(sf->fptr, TFLOAT,  7,  row, 1, 1, &(dcols->elevation), status);
    fits_write_col(sf->fptr, TFLOAT,  8,  row, 1, 1, &(dcols->bmaj), status);
    fits_write_col(sf->fptr, TFLOAT,  9,  row, 1, 1, &(dcols->bmin), status);
    fits_write_col(sf->fptr, TFLOAT,  10,  row, 1, 1, &(dcols->bpa), status);
    fits_write_col(sf->fptr, TINT,    11,  row, 1, 1, &(dcols->accumid), status);
    fits_write_col(sf->fptr, TINT,    12, row, 1, 1, &(dcols->sttspec), status);
    fits_write_col(sf->fptr, TINT,    13, row, 1, 1, &(dcols->stpspec), status);
    fits_write_col(sf->fptr, TDOUBLE, 14, row, 1, nsubband, dcols->centre_freq, status);

    fits_write_col(sf->fptr, TFLOAT,  15, row, 1, 1, &(dcols->centre_freq_idx), status);
    fits_write_col(sf->fptr, TDOUBLE, 16, row, 1, 1, &temp_dbl, status);
    fits_write_col(sf->fptr, TDOUBLE, 17, row, 1, 1, &(sf->hdr.chan_bw), status);
    fits_write_col(sf->fptr, TDOUBLE, 18, row, 1, 1, &(dcols->ra), status);
    fits_write_col(sf->fptr, TDOUBLE, 19, row, 1, 1, &(dcols->dec), status);
    fits_write_col(sf->fptr, TFLOAT,  20, row, 1, nivals, dcols->data, status);

    // Flush the buffers if not finished with the file
    // Note:  this use is not entirely in keeping with the CFITSIO
    //        documentation recommendations.  However, manually 
    //        correcting NAXIS2 and using fits_flush_buffer()
    //        caused occasional hangs (and extrememly large
    //        files due to some infinite loop).
    fits_flush_file(sf->fptr, status);
    // Print status if bad
    if (*status) {
        fprintf(stderr, "Error writing subint %d:\n", sf->rownum);
        fits_report_error(stderr, *status);
        fflush(stderr);
    }

    // Now update some key values if no CFITSIO errors
    if (!(*status)) {
        sf->rownum++;
        sf->tot_rows++;
        sf->N += 1;
        sf->T += dcols->exposure;
    }

    return *status;
}


int sdfits_close(struct sdfits *sf) {
    if (!sf->status) {
        fits_close_file(sf->fptr, &(sf->status));
        printf("Closing file '%s'\n", sf->filename);
    }
    printf("Done.  %s %d data rows (%f sec) in %d files (status = %d).\n",
            sf->mode=='r' ? "Read" : "Wrote", 
            sf->tot_rows, sf->T, sf->filenum, sf->status);
    return sf->status;
}
