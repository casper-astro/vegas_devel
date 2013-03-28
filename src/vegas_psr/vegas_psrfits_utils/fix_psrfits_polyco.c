/* fix_psrfits_polyco.c
 *
 * Install missing polycos into a psrfits file.
 */ 
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <getopt.h>
#include "polyco.h"
#include "psrfits.h"

int main(int argc, char *argv[]) {

    /* cmd line */
    static struct option long_opts[] = {
        {"parfile", 1, NULL, 'P'},
        {0,0,0,0}
    };
    int opt, opti;
    char parfile[256]="";
    while ((opt=getopt_long(argc,argv,"P:",long_opts,&opti))!=-1) {
        switch (opt) {
            case 'P':
                strncpy(parfile, optarg, 255);
                parfile[255]='\0';
                break;
            default:
                exit(0);
                break;
        }
    }
    if (optind==argc) {
        fprintf(stderr, "No files given.\n");
        exit(1);
    }
    if (parfile[0]=='\0') {
        fprintf(stderr, "No .par file given.\n");
        exit(1);
    }

    int i, rv;
    struct psrfits pf;
    struct polyco *pc=NULL;
    for (i=optind; i<argc; i++) {

        // Open file
        strcpy(pf.filename, argv[i]);
        pf.status=0;
        rv = psrfits_open(&pf);
        if (rv) { fits_report_error(stderr, rv); exit(1); }

        // Basic info
        int nsubint = pf.rows_per_file;
        printf("Found %d subints.\n", nsubint);

        // Make polycos
        char source[32];
        int npc = make_polycos(parfile, &pf.hdr, source, &pc);
        if (npc<=0) {
            fprintf(stderr, "Error generating polycos.\n");
            exit(1);
        }

        // Decide which ones are needed 
        int isub, col;
        fits_movnam_hdu(pf.fptr, BINARY_TBL, "SUBINT", 0, &pf.status);
        fits_get_colnum(pf.fptr, CASEINSEN, "OFFS_SUB", &col, &pf.status);
        for (isub=0; isub<nsubint; isub++) {
            double offset, fmjd;
            fits_read_col(pf.fptr, TDOUBLE, col, isub+1, 1, 1, NULL, &offset,
                    NULL, &pf.status);
            fmjd = (pf.hdr.start_sec + offset)/86400.0;
            int ipc = select_pc(pc, npc, source, pf.hdr.start_day, fmjd);
            if (ipc<0) { 
                fprintf(stderr, "Polycos do not span observation range.\n");
                exit(1);
            }
            printf("Polyco set %d was used (isub=%d offset=%f)\n", 
                    ipc, isub, offset);
            pc[ipc].used = 1;
        }

        // Close file
        psrfits_close(&pf);

        // Reopen for writing
        rv = fits_open_file(&pf.fptr, pf.filename, READWRITE, &pf.status);
        if (rv) { fits_report_error(stderr, rv); exit(1); }

        // Delete all old polycos??????
        long nrows;
        fits_movnam_hdu(pf.fptr, BINARY_TBL, "POLYCO", 0, &pf.status);
        fits_get_num_rows(pf.fptr, &nrows, &pf.status);
        fits_delete_rows(pf.fptr, 1, nrows, &pf.status);

        // Fill in new polycos
        rv = psrfits_write_polycos(&pf, pc, npc);
        if (rv) { fits_report_error(stderr, pf.status);  pf.status=0; }

        // Close file, all done
        fits_close_file(pf.fptr, &pf.status);
    }

    exit(0);

}
