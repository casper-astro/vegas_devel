/* check_guppi_status.c
 *
 * Basic prog to test status shared mem routines.
 */
#include <stdio.h>
#include <stdlib.h>
#include <getopt.h>

#include "fitshead.h"
#include "guppi_error.h"
#include "guppi_status.h"

int main(int argc, char *argv[]) {

    int rv;
    struct guppi_status s;

    rv = guppi_status_attach(&s);
    if (rv!=GUPPI_OK) {
        fprintf(stderr, "Error connecting to shared mem.\n");
        perror(NULL);
        exit(1);
    }

    guppi_status_lock(&s);

    /* Loop over cmd line to fill in params */
    static struct option long_opts[] = {
        {"key",    1, NULL, 'k'},
        {"get",    1, NULL, 'g'},
        {"string", 1, NULL, 's'},
        {"float",  1, NULL, 'f'},
        {"double", 1, NULL, 'd'},
        {"int",    1, NULL, 'i'},
        {"quiet",  0, NULL, 'q'},
        {"clear",  0, NULL, 'C'},
        {"del",    0, NULL, 'D'},
        {0,0,0,0}
    };
    int opt,opti;
    char *key=NULL;
    float flttmp;
    double dbltmp;
    int inttmp;
    int quiet=0, clear=0;
    while ((opt=getopt_long(argc,argv,"k:g:s:f:d:i:qCD",long_opts,&opti))!=-1) {
        switch (opt) {
            case 'k':
                key = optarg;
                break;
            case 'g':
                hgetr8(s.buf, optarg, &dbltmp);
                printf("%g\n", dbltmp);
                break;
            case 's':
                if (key) 
                    hputs(s.buf, key, optarg);
                break;
            case 'f':
                flttmp = atof(optarg);
                if (key) 
                    hputr4(s.buf, key, flttmp);
                break;
            case 'd':
                dbltmp = atof(optarg);
                if (key) 
                    hputr8(s.buf, key, dbltmp);
                break;
            case 'i':
                inttmp = atoi(optarg);
                if (key) 
                    hputi4(s.buf, key, inttmp);
                break;
            case 'D':
                if (key)
                    hdel(s.buf, key);
                break;
            case 'C':
                clear=1;
                break;
            case 'q':
                quiet=1;
                break;
            default:
                break;
        }
    }

    /* If not quiet, print out buffer */
    if (!quiet) { 
        printf(s.buf); printf("\n"); 
    }

    guppi_status_unlock(&s);

    if (clear) 
        guppi_status_clear(&s);

    exit(0);
}
