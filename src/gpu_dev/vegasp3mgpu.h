/* 
 * vegasp3mgpu.h
 * VEGAS Priority 3 Mode - Stand-Alone GPU Implementation
 *
 * Created by Jayanth Chennamangalam on 2011.06.02
 */

#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <cufft.h>

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
#include <math.h>       /* for ceilf(), and log10f() in Plot() */
#include <sys/time.h>   /* for gettimeofday() */

#define COPY                1
#define CONSTMEM            0

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

#if PLOT
/* PGPLOT macro definitions */
#define PG_DEV              "1/XS"
#define PG_VP_ML            0.17    /* left margin */
#define PG_VP_MR            0.83    /* right margin */
#define PG_VP_MB            0.17    /* bottom margin */
#define PG_VP_MT            0.83    /* top margin */
#define PG_SYMBOL           2
#define PG_CI_DEF           1
#define PG_CI_PLOT          11
#endif

typedef unsigned char BYTE;

int Init(void);
int LoadData(void);
int ReadData(void);
__global__ void Convert(float *pfConvLookup, BYTE *pbData, float *pfData);
__global__ void Unpack(cufftComplex *pccDataX, cufftComplex *pccDataY, float *pfData);
#if CONSTMEM
__global__ void DoPFB(cufftComplex *pccDataX,
                      cufftComplex *pccDataY,
                      int iPFBReadIdx,
                      int iNTaps,
                      cufftComplex *pccFFTInX,
                      cufftComplex *pccFFTInY);
#else
__global__ void DoPFB(cufftComplex *pccDataX,
                      cufftComplex *pccDataY,
                      int iPFBReadIdx,
                      int iNTaps,
                      float *pfPFBCoeff,
                      cufftComplex *pccFFTInX,
                      cufftComplex *pccFFTInY);
#endif
#if (!(COPY))
int CopyDataForFFT(void);
#else
__global__ void CopyDataForFFT(cufftComplex *pccFFTInX,
                               cufftComplex *pccFFTInY,
                               cufftComplex *pccDataX,
                               cufftComplex *pccDataY,
                               int iNFFT);
#endif
int DoFFT(void);
int IsRunning(void);
int IsBlankingSet(void);
void CleanUp(void);

#define VEGASCUDASafeCall(iRet)     __VEGASCUDASafeCall(iRet,                 \
                                                        __FILE__,             \
                                                        __LINE__,             \
                                                        &CleanUp)

void __VEGASCUDASafeCall(cudaError_t iRet,
                         const char* pcFile,
                         const int iLine,
                         void (*pCleanUp)(void));

#if PLOT
/* PGPLOT function declarations */
void InitPlot(void);
void Plot(void);
#endif

int RegisterSignalHandlers();
void HandleStopSignals(int iSigNo);

void PrintUsage(const char *pcProgName);

