#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "fitsio.h"

// This tests to see if 2 times are within 100ns of each other
#define TEST_CLOSE(a, b) (fabs((a)-(b)) <= 1e-7 ? 1 : 0)

int main(int argc, char *argv[]) {
    fitsfile *pf;
    double offs, last_offs, diff_offs, num_blocks, row_duration;
    int ii, firsttime = 1, filenum = 1, status = 0, anynull = 0;
    long nrows, row = 1;
    char filename[100];

    while (1) {
        sprintf(filename, "%s_%04d.fits", argv[1], filenum);
        fits_open_file(&pf, filename, READONLY, &status);
        if (status) break;
        fits_movabs_hdu(pf, 2, NULL, &status);
        fits_get_num_rows(pf, &nrows, &status);
        printf("Working on '%s' with %ld rows\n", filename, nrows);
        row = 1;
        for (ii = 1 ; ii <= nrows ; ii++) {
            if (firsttime) {
                fits_read_col(pf, TDOUBLE, 1, row, 1, 1,
                              0, &row_duration, &anynull, &status);
                last_offs = -0.5*row_duration;
                firsttime = 0;
            }
            fits_read_col(pf, TDOUBLE, 2, row, 1, 1,
                          0, &offs, &anynull, &status);
            diff_offs = offs - last_offs;
            if (!TEST_CLOSE(diff_offs, row_duration)) {
                num_blocks = diff_offs/row_duration;
                printf("At row %ld, found %.3f dropped rows.\n", 
                       row, num_blocks);
            }
            last_offs = offs;
            row++;
        }
        filenum++;
        printf("  checked %d rows.\n", row-1);
        fits_close_file(pf, &status);
    }
    if (status != 104)  // 104 is file could not be opened or some such
        printf("Exited with status = %d\n", status);
    exit(0);
}
