/**
 * @file vegas_gentestdata.c
 * Program to generate test data for the VEGAS standalone implementations. The
 *  test data is made up of 1-byte signed values in the range -128 to 127 that
 *  may be interpreted to be 8_7 fixed-point values in the range
 *  [-1.0, 0.992188]. The VEGAS standalone implementations treat this data as
 *  interleaved, complex, dual-polarisation data, with an arbitrary number of
 *  sub-bands.
 *
 * @author Jayanth Chennamangalam
 * @date 2011.08.01
 */

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <float.h>
#include <getopt.h>
#include <string.h>
#include <errno.h>
#include <assert.h>

#define LEN_GENSTRING   256
#define DEF_NUM_SAMPS   10485760    /* 10M samples */
#define F_S             128.0       /* sampling frequency in MHz */
#define NUM_FREQS       4
#define NUM_SUBBANDS    1
#define SCALE_FACTOR    127

void PrintUsage(const char *pcProgName);

int main(int argc, char *argv[])
{
    signed char cDataReX = 0;
    signed char cDataImX = 0;
    signed char cDataReY = 0;
    signed char cDataImY = 0;
    int i = 0;
    int j = 0;
    int iLen = DEF_NUM_SAMPS;
    int iFile = 0;
    char acFileData[LEN_GENSTRING] = {0};
    float afFreqX[NUM_FREQS] = {10.0, 25.125, 70.2, 48.0};  /* in MHz */
    float afFreqY[NUM_FREQS] = {10.0, 25.125, 70.2, 48.0};  /* in MHz */
    const char *pcProgName = NULL;
    int iNextOpt = 0;
    /* valid short options */
    const char* const pcOptsShort = "hn:";
    /* valid long options */
    const struct option stOptsLong[] = {
        { "help",           0, NULL, 'h' },
        { "nsamp",          1, NULL, 'n' },
        { NULL,             0, NULL, 0   }
    };

    /* get the filename of the program from the argument list */
    pcProgName = argv[0];

    /* parse the input */
    do
    {
        iNextOpt = getopt_long(argc, argv, pcOptsShort, stOptsLong, NULL);
        switch (iNextOpt)
        {
            case 'h':   /* -h or --help */
                /* print usage info and terminate */
                PrintUsage(pcProgName);
                return EXIT_SUCCESS;

            case 'n':   /* -n or --nsamp */
                /* set option */
                iLen = (int) atoi(optarg);
                break;

            case '?':   /* user specified an invalid option */
                /* print usage info and terminate with error */
                (void) fprintf(stderr, "ERROR: Invalid option!\n");
                PrintUsage(pcProgName);
                return EXIT_FAILURE;

            case -1:    /* done with options */
                break;

            default:    /* unexpected */
                assert(0);
        }
    } while (iNextOpt != -1);

    /* no arguments */
    if (argc <= optind)
    {
        (void) fprintf(stderr,
                       "ERROR: Data file not specified!\n");
        PrintUsage(pcProgName);
        return EXIT_FAILURE;
    }

    (void) strncpy(acFileData, argv[optind], LEN_GENSTRING);
    acFileData[LEN_GENSTRING-1] = '\0';

    iFile = open(acFileData,
                 O_CREAT | O_TRUNC | O_WRONLY,
                 S_IRUSR | S_IWUSR | S_IRGRP | S_IROTH);
    if (EXIT_FAILURE == iFile)
    {
        (void) fprintf(stderr,
                       "ERROR: Opening file failed! %s.\n",
                       strerror(errno));
        return EXIT_FAILURE;
    }

    srand((unsigned int) time(NULL));
    for (i = 0; i < iLen; ++i)
    {
        /* generate data - pol X */
        /* add some noise */
        cDataReX = SCALE_FACTOR * ((0.6 * ((double) rand()) / RAND_MAX) - 0.3);
        cDataImX = SCALE_FACTOR * ((0.6 * ((double) rand()) / RAND_MAX) - 0.3);
        /* add some signal */
        for (j = 0; j < NUM_FREQS; ++j)
        {
            /* NOTE: make sure that the sum is within [-128:127] */
            cDataReX += SCALE_FACTOR
                        * (0.1 * cos(2 * M_PI * afFreqX[j] * i / F_S));
            cDataImX += SCALE_FACTOR
                        * (0.1 * sin(2 * M_PI * afFreqX[j] * i / F_S));
        }

        /* write to disk */
        (void) write(iFile, &cDataReX, sizeof(signed char));
        (void) write(iFile, &cDataImX, sizeof(signed char));

        /* generate data - pol Y */
        /* add some noise */
        cDataReY = SCALE_FACTOR * ((0.2 * ((double) rand()) / RAND_MAX) - 0.1);
        cDataImY = SCALE_FACTOR * ((0.2 * ((double) rand()) / RAND_MAX) - 0.1);
        /* add some signal */
        for (j = 0; j < NUM_FREQS; ++j)
        {
            /* NOTE: make sure that the sum is within [-128:127] */
            cDataReY += SCALE_FACTOR
                        * (0.2 * cos(2 * M_PI * afFreqY[j] * i / F_S));
            cDataImY += SCALE_FACTOR
                        * (0.2 * sin(2 * M_PI * afFreqY[j] * i / F_S));
        }

        /* write to disk */
        (void) write(iFile, &cDataReY, sizeof(signed char));
        (void) write(iFile, &cDataImY, sizeof(signed char));
    }

    (void) close(iFile);

    return EXIT_SUCCESS;
}

/**
 * Prints usage information
 */
void PrintUsage(const char *pcProgName)
{
    (void) printf("Usage: %s [options] <data-file>\n",
                  pcProgName);
    (void) printf("    -h  --help                           ");
    (void) printf("Display this usage information\n");
    (void) printf("    -n  --nsamp <value>                  ");
    (void) printf("Number of time samples\n");

    return;
}

