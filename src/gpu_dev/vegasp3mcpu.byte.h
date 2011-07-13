/* 
 * vegasp3mcpu.byte.h
 * VEGAS Priority 3 Mode - Stand-Alone CPU Implementation
 *
 * Created by Jayanth Chennamangalam on 2011.06.02
 */

#include <stdio.h>
#include <stdlib.h>
#include <fftw3.h>

#define PLOT                0

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

#define OUTFILE             0

#define FALSE               0
#define TRUE                1

#define DEF_PFB_ON          FALSE

#define NUM_BYTES_PER_SAMP  4
#define DEF_LEN_SPEC        1024    /* default value for g_iNFFT */

#define LEN_DATA            (NUM_BYTES_PER_SAMP * g_iNFFT)

#define DEF_ACC             1       /* default number of spectra to
                                       accumulate */
/* GUPPI defs */
#define GUPPI_OK            0
#define GUPPI_ERR_GEN       -1

/* for PFB */
#define NUM_TAPS            8       /* number of multiples of g_iNFFT */
#define FILE_COEFF_PREFIX   "tests/python/coeff_"
#define FILE_COEFF_SUFFIX   ".dat"

#define USEC2SEC            1e-6

/* PGPLOT macro definitions */
#define PG_DEV              "1/XS"
#define PG_VP_ML            0.17    /* left margin */
#define PG_VP_MR            0.83    /* right margin */
#define PG_VP_MB            0.17    /* bottom margin */
#define PG_VP_MT            0.83    /* top margin */
#define PG_SYMBOL           2
#define PG_CI_DEF           1
#define PG_CI_PLOT          11

typedef unsigned char BYTE;

typedef struct tagPFBData
{
    BYTE *pbData;               /* raw data, LEN_DATA long*/
    signed char (*pabDataX)[][2];       /* unpacked pol-X data, g_iNFFT long */
    signed char (*pabDataY)[][2];       /* unpacked pol-Y data, g_iNFFT long */
    int iNextIdx;               /* index of next element in PFB ring buffer */
} PFB_DATA;

int Init(void);
int LoadData(void);
int ReadData(void);
int DoPFB(void);
int CopyDataForFFT(void);
int DoFFT(void);
int IsRunning(void);
int IsBlankingSet(void);
void CleanUp(void);

#if PLOT
/* PGPLOT function declarations */
void InitPlot(void);
void Plot(void);
#endif

int RegisterSignalHandlers();
void HandleStopSignals(int iSigNo);

void PrintUsage(const char *pcProgName);

