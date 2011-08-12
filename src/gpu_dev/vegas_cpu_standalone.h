/* 
 * @file vegas_cpu_standalone.h
 * VEGAS Low-Bandwidth Modes - Stand-Alone CPU Implementation
 *
 * @author Jayanth Chennamangalam
 * @date 2011.06.02
 */

#ifndef __VEGAS_CPU_STANDALONE_H__
#define __VEGAS_CPU_STANDALONE_H__

#include <stdio.h>
#include <stdlib.h>
#include <fftw3.h>

#define PLOT                1
#define OUTFILE             0

#include <string.h>     /* for memset(), strncpy(), memcpy(), strerror() */
#include <sys/types.h>  /* for open() */
#include <sys/stat.h>   /* for open() */
#include <fcntl.h>      /* for open() */
#include <unistd.h>     /* for close() and usleep() */
#if PLOT
#include <cpgplot.h>    /* for cpg*() */
#endif
#include <float.h>      /* for FLT_MAX */
#include <getopt.h>     /* for option parsing */
#include <assert.h>     /* for assert() */
#include <errno.h>      /* for errno */
#include <signal.h>     /* for signal-handling */
#include <math.h>       /* for log10f() in Plot() */
#include <sys/time.h>   /* for gettimeofday() */

#define FALSE               0
#define TRUE                1

#define DEF_PFB_ON          FALSE

#define NUM_BYTES_PER_SAMP  4
#define DEF_LEN_SPEC        1024        /* default value for g_iNFFT */

#define DEF_SIZE_READ       33554432    /* 32 MB - block size in VEGAS input
                                           buffer */
#define LEN_DATA            (NUM_BYTES_PER_SAMP * g_iNFFT)

#define DEF_ACC             1           /* default number of spectra to
                                           accumulate */
/* for PFB */
#define NUM_TAPS            8       /* number of multiples of g_iNFFT */
#define FILE_COEFF_PREFIX   "coeff"
#define FILE_COEFF_DATATYPE "signedchar"
#define FILE_COEFF_SUFFIX   ".dat"

#define DEF_NUM_SUBBANDS    1       /* NOTE: no support for > 1 */

#define USEC2SEC            1e-6

#if PLOT
/* PGPLOT macro definitions */
#define PG_DEV              "1/XS"
#define PG_VP_ML            0.10    /* left margin */
#define PG_VP_MR            0.90    /* right margin */
#define PG_VP_MB            0.12    /* bottom margin */
#define PG_VP_MT            0.98    /* top margin */
#define PG_SYMBOL           2
#define PG_CI_DEF           1
#define PG_CI_PLOT          11
#endif

typedef struct tagPFBData
{
    signed char *pcData;        /* raw data, LEN_DATA long*/
    fftwf_complex *pfcDataX;    /* unpacked pol-X data, g_iNFFT long */
    fftwf_complex *pfcDataY;    /* unpacked pol-Y data, g_iNFFT long */
    int iNextIdx;               /* index of next element in PFB ring buffer */
} PFB_DATA;

/**
 * Initialises the program.
 */
int Init(void);

/**
 * Reads all data from the input file and loads it into memory.
 */
int LoadDataToMem(void);

/**
 * Reads one block (32MB) of data form memory.
 */
int ReadData(void);

/*
 * Perform polyphase filtering.
 */
int DoPFB(void);
int CopyDataForFFT(void);
int DoFFT(void);
void CleanUp(void);

#if PLOT
/* PGPLOT function declarations */
int InitPlot(void);
void Plot(void);
#endif

int RegisterSignalHandlers();
void HandleStopSignals(int iSigNo);
void PrintUsage(const char* pcProgName);

#endif  /* __VEGAS_CPU_STANDALONE_H__ */

