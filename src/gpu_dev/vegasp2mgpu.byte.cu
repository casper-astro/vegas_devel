/* 
 * vegasp2mgpu.byte.cu
 * VEGAS Priority 2 Mode - Stand-Alone GPU Implementation
 *
 * Created by Jayanth Chennamangalam on 2011.06.13
 */

#include "vegasp2mgpu.byte.h"

#define BENCHMARKING    0

int g_iIsDone = FALSE;

int g_iMaxThreadsPerBlock = 0;

BYTE *g_pbInBuf = NULL;
BYTE *g_pbInBufRead = NULL;
int g_iReadCount = 0;
int g_iNumReads = 0;

BYTE* g_apbData_d[NUM_TAPS] = {NULL};   /* raw data, LEN_DATA long*/
signed char (*g_pabDataX_d)[][2*NUM_SUBBANDS] = NULL;/* unpacked pol-X data, g_iNFFT long */
signed char (*g_pabDataY_d)[][2*NUM_SUBBANDS] = NULL;/* unpacked pol-Y data, g_iNFFT long */

int g_iPFBReadIdx = 0;
int g_iPFBWriteIdx = 0;

int g_iNFFT = DEF_LEN_SPEC;

dim3 g_dimBlockConv(1, 1, 1);
dim3 g_dimGridConv(1, 1);
dim3 g_dimBlockUnpack(1, 1, 1);
dim3 g_dimGridUnpack(1, 1);
dim3 g_dimBlockPFB(1, 1, 1);
dim3 g_dimGridPFB(1, 1);
dim3 g_dimBlockCopy(1, 1, 1);
dim3 g_dimGridCopy(1, 1);
dim3 g_dimBlockAccum(1, 1, 1);
dim3 g_dimGridAccum(1, 1);

cufftComplex *g_pccFFTInX = NULL;
cufftComplex *g_pccFFTInX_d = NULL;
cufftComplex *g_pccFFTOutX = NULL;
cufftComplex *g_pccFFTOutX_d = NULL;
cufftHandle g_stPlanX = {0};
cufftComplex *g_pccFFTInY = NULL;
cufftComplex *g_pccFFTInY_d = NULL;
cufftComplex *g_pccFFTOutY = NULL;
cufftComplex *g_pccFFTOutY_d = NULL;
cufftHandle g_stPlanY = {0};

float *g_pfSumPowX = NULL;
float *g_pfSumPowY = NULL;
float *g_pfSumStokesRe = NULL;
float *g_pfSumStokesIm = NULL;

float *g_pfSumPowX_d = NULL;
float *g_pfSumPowY_d = NULL;
float *g_pfSumStokesRe_d = NULL;
float *g_pfSumStokesIm_d = NULL;

int g_iIsPFBOn = DEF_PFB_ON;
int g_iNTaps = 1;                       /* 1 if no PFB, NUM_TAPS if PFB */
int g_iFileCoeff = 0;
char g_acFileCoeff[256] = {0};
signed char *g_pcPFBCoeff = NULL;
signed char *g_pcPFBCoeff_d = NULL;

float g_afConvLookup[256] = {0};
float *g_pfConvLookup_d = NULL;

int g_iFileData = 0;
char g_acFileData[256] = {0};

/* PGPLOT global */
float *g_pfFreq = NULL;
float g_fFSamp = 1.0;                   /* 1 [frequency] */

#if BENCHMARKING
    float g_fTimeCpIn = 0.0;
    float g_fAvgCpIn = 0.0;
    float g_fTimeUnpack = 0.0;
    float g_fAvgUnpack = 0.0;
    cudaEvent_t g_cuStart;
    cudaEvent_t g_cuStop;
    int g_iCount = 0;
#endif

int main(int argc, char *argv[])
{
    int iRet = GUPPI_OK;
    int iTime = 0;
    int iAcc = DEF_ACC;
#if BENCHMARKING
    float fTimePFB = 0.0;
    float fAvgPFB = 0.0;
    float fTimeCpInFFT = 0.0;
    float fAvgCpInFFT = 0.0;
    float fTimeFFT = 0.0;
    float fAvgFFT = 0.0;
    float fTimeCpOut = 0.0;
    float fAvgCpOut = 0.0;
    float fTimeAccum = 0.0;
    float fAvgAccum = 0.0;
    float fAvgTotal = 0.0;
#else
    struct timeval stStart = {0};
    struct timeval stStop = {0};
#endif
    const char *pcProgName = NULL;
    int iNextOpt = 0;
    /* valid short options */
    const char* const pcOptsShort = "hn:pa:s:";
    /* valid long options */
    const struct option stOptsLong[] = {
        { "help",           0, NULL, 'h' },
        { "nfft",           1, NULL, 'n' },
        { "pfb",            0, NULL, 'p' },
        { "nacc",           1, NULL, 'a' },
        { "fsamp",          1, NULL, 's' },
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

            case 'n':   /* -n or --nfft */
                /* set option */
                g_iNFFT = (int) atoi(optarg);
                break;

            case 'p':   /* -p or --pfb */
                /* set option */
                g_iIsPFBOn = TRUE;
                break;

            case 'a':   /* -a or --nacc */
                /* set option */
                iAcc = (int) atoi(optarg);
                break;

            case 's':   /* -s or --fsamp */
                /* set option */
                g_fFSamp = (float) atof(optarg);
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
        (void) fprintf(stderr, "ERROR: Data file not specified!\n");
        PrintUsage(pcProgName);
        return GUPPI_ERR_GEN;
    }

    (void) strncpy(g_acFileData, argv[optind], 256);
    g_acFileData[255] = '\0';

    /* initialise */
    iRet = Init();
    if (iRet != GUPPI_OK)
    {
        (void) fprintf(stderr, "ERROR! Init failed!\n");
        CleanUp();
        return GUPPI_ERR_GEN;
    }

#if BENCHMARKING
    (void) printf("* Benchmarking run commencing...\n");
    VEGASCUDASafeCall(cudaEventCreate(&g_cuStart));
    VEGASCUDASafeCall(cudaEventCreate(&g_cuStop));
    (void) printf("* Events created.\n");
#else
    (void) gettimeofday(&stStart, NULL);
#endif
    while (IsRunning())
    {
#if BENCHMARKING
        ++g_iCount;
#endif
        if (g_iIsPFBOn)
        {
            /* do pfb */
#if BENCHMARKING
            VEGASCUDASafeCall(cudaEventRecord(g_cuStart, 0));
            VEGASCUDASafeCall(cudaEventSynchronize(g_cuStart));
#endif
            DoPFB<<<g_dimGridPFB, g_dimBlockPFB>>>((signed char *) g_pabDataX_d,
                                                   (signed char *) g_pabDataY_d,
                                                   g_iPFBReadIdx,
                                                   g_iNTaps,
                                                   g_pcPFBCoeff_d,
                                                   g_pccFFTInX_d,
                                                   g_pccFFTInY_d);
            VEGASCUDASafeCall(cudaThreadSynchronize());
#if BENCHMARKING
            VEGASCUDASafeCall(cudaEventRecord(g_cuStop, 0));
            VEGASCUDASafeCall(cudaEventSynchronize(g_cuStop));
            VEGASCUDASafeCall(cudaEventElapsedTime(&fTimePFB, g_cuStart, g_cuStop));
            fAvgPFB = (fTimePFB + ((g_iCount - 1) * fAvgPFB)) / g_iCount;
#endif
        }
        else
        {
#if BENCHMARKING
            VEGASCUDASafeCall(cudaEventRecord(g_cuStart, 0));
            VEGASCUDASafeCall(cudaEventSynchronize(g_cuStart));
#endif
            CopyDataForFFT<<<g_dimGridCopy, g_dimBlockCopy>>>(g_pccFFTInX_d,
                                                              g_pccFFTInY_d,
                                                              ((signed char *) g_pabDataX_d) + (2 * g_iPFBReadIdx * g_iNFFT),
                                                              ((signed char *) g_pabDataY_d) + (2 * g_iPFBReadIdx * g_iNFFT),
                                                              g_iNFFT,
                                                              g_pfConvLookup_d);
            VEGASCUDASafeCall(cudaThreadSynchronize());
#if BENCHMARKING
            VEGASCUDASafeCall(cudaEventRecord(g_cuStop, 0));
            VEGASCUDASafeCall(cudaEventSynchronize(g_cuStop));
            VEGASCUDASafeCall(cudaEventElapsedTime(&fTimeCpInFFT, g_cuStart, g_cuStop));
            fAvgCpInFFT = (fTimeCpInFFT + ((g_iCount - 1) * fAvgCpInFFT)) / g_iCount;
#endif
        }

        /* do fft */
#if BENCHMARKING
        VEGASCUDASafeCall(cudaEventRecord(g_cuStart, 0));
        VEGASCUDASafeCall(cudaEventSynchronize(g_cuStart));
#endif
        (void) DoFFT();

#if BENCHMARKING
        VEGASCUDASafeCall(cudaEventRecord(g_cuStop, 0));
        VEGASCUDASafeCall(cudaEventSynchronize(g_cuStop));
        VEGASCUDASafeCall(cudaEventElapsedTime(&fTimeFFT, g_cuStart, g_cuStop));
        fAvgFFT = (fTimeFFT + ((g_iCount - 1) * fAvgFFT)) / g_iCount;
#endif

        /* accumulate power x, power y, stokes, if the blanking bit is
           not set */
        if (!IsBlankingSet())
        {
            if (0/* blanking to non-blanking */)
            {
                /* TODO: when blanking is unset, start accumulating */
                /* reset time */
                iTime = 0;
                /* zero accumulators */
                (void) memset(g_pfSumPowX, '\0', g_iNFFT * sizeof(float));
                (void) memset(g_pfSumPowY, '\0', g_iNFFT * sizeof(float));
                (void) memset(g_pfSumStokesRe, '\0', g_iNFFT * sizeof(float));
                (void) memset(g_pfSumStokesIm, '\0', g_iNFFT * sizeof(float));
            }
            else
            {
#if BENCHMARKING
                VEGASCUDASafeCall(cudaEventRecord(g_cuStart, 0));
                VEGASCUDASafeCall(cudaEventSynchronize(g_cuStart));
#endif
                Accumulate<<<g_dimGridAccum, g_dimBlockAccum>>>(g_pccFFTOutX_d,
                                                                g_pccFFTOutY_d,
                                                                g_pfSumPowX_d,
                                                                g_pfSumPowY_d,
                                                                g_pfSumStokesRe_d,
                                                                g_pfSumStokesIm_d);
                VEGASCUDASafeCall(cudaThreadSynchronize());
#if BENCHMARKING
                VEGASCUDASafeCall(cudaEventRecord(g_cuStop, 0));
                VEGASCUDASafeCall(cudaEventSynchronize(g_cuStop));
                VEGASCUDASafeCall(cudaEventElapsedTime(&fTimeAccum, g_cuStart, g_cuStop));
                fAvgAccum = (fTimeAccum + ((g_iCount - 1) * fAvgAccum)) / g_iCount;
#endif
                ++iTime;
                if (iTime == iAcc)
                {
                    #if PLOT
                    /* NOTE: Plot() will modify data! */
                    Plot();
                    usleep(500000);
                    #endif

                    /* dump to buffer */
#if BENCHMARKING
                    VEGASCUDASafeCall(cudaEventRecord(g_cuStart, 0));
                    VEGASCUDASafeCall(cudaEventSynchronize(g_cuStart));
#endif
                    VEGASCUDASafeCall(cudaMemcpy(g_pfSumPowX,
                                                 g_pfSumPowX_d,
                                                 g_iNFFT * sizeof(float),
                                                 cudaMemcpyDeviceToHost));
                    VEGASCUDASafeCall(cudaMemcpy(g_pfSumPowY,
                                                 g_pfSumPowY_d,
                                                 g_iNFFT * sizeof(float),
                                                 cudaMemcpyDeviceToHost));
                    VEGASCUDASafeCall(cudaMemcpy(g_pfSumStokesRe,
                                                 g_pfSumStokesRe_d,
                                                 g_iNFFT * sizeof(float),
                                                 cudaMemcpyDeviceToHost));
                    VEGASCUDASafeCall(cudaMemcpy(g_pfSumStokesIm,
                                                 g_pfSumStokesIm_d,
                                                 g_iNFFT * sizeof(float),
                                                 cudaMemcpyDeviceToHost));
#if BENCHMARKING
                    VEGASCUDASafeCall(cudaEventRecord(g_cuStop, 0));
                    VEGASCUDASafeCall(cudaEventSynchronize(g_cuStop));
                    VEGASCUDASafeCall(cudaEventElapsedTime(&fTimeCpOut, g_cuStart, g_cuStop));
                    fAvgCpOut = (fTimeCpOut + ((g_iCount - 1) * fAvgCpOut)) / g_iCount;
#endif

                    /* reset time */
                    iTime = 0;
                    /* zero accumulators */
                    VEGASCUDASafeCall(cudaMemset(g_pfSumPowX_d, '\0', g_iNFFT * sizeof(float)));
                    VEGASCUDASafeCall(cudaMemset(g_pfSumPowY_d, '\0', g_iNFFT * sizeof(float)));
                    VEGASCUDASafeCall(cudaMemset(g_pfSumStokesRe_d, '\0', g_iNFFT * sizeof(float)));
                    VEGASCUDASafeCall(cudaMemset(g_pfSumStokesIm_d, '\0', g_iNFFT * sizeof(float)));
                }
            }
        }
        else
        {
            /* TODO: */
            if (1/* non-blanking to blanking */)
            {
                /* write status, dump data to disk buffer */
            }
            else
            {
                /* do nothing, wait for blanking to stop */
            }
        }

        /* read data from input buffer, convert 8_7 to float */
        iRet = ReadData();
        if (iRet != GUPPI_OK)
        {
            (void) fprintf(stderr, "ERROR: Data reading failed!\n");
            break;
        }
    }
#if (!BENCHMARKING)
    (void) gettimeofday(&stStop, NULL);
    (void) printf("Time taken (barring Init()): %gs\n",
                  ((stStop.tv_sec + (stStop.tv_usec * USEC2SEC))
                   - (stStart.tv_sec + (stStart.tv_usec * USEC2SEC))));
#endif

    CleanUp();

#if BENCHMARKING
    fAvgTotal = g_fAvgCpIn + g_fAvgUnpack + fAvgPFB + fAvgCpInFFT + fAvgFFT + fAvgAccum + fAvgCpOut;
    (void) printf("    Average elapsed time for %d\n", g_iCount);
    (void) printf("        calls to cudaMemcpy(Host2Device)          : %5.3fms, %2d%%\n",
                  g_fAvgCpIn,
                  (int) ((g_fAvgCpIn / fAvgTotal) * 100));
    (void) printf("        calls to Unpack()                         : %5.3fms, %2d%%\n",
                  g_fAvgUnpack,
                  (int) ((g_fAvgUnpack / fAvgTotal) * 100));
    if (g_iIsPFBOn)
    {
        (void) printf("        calls to DoPFB()                          : %5.3fms, %2d%%\n",
                      fAvgPFB,
                      (int) ((fAvgPFB / fAvgTotal) * 100));
    }
    else
    {
        (void) printf("        calls to CopyDataForFFT()                 : %5.3fms, %2d%%\n",
                      fAvgCpInFFT,
                      (int) ((fAvgCpInFFT / fAvgTotal) * 100));
    }
    (void) printf("        calls to DoFFT()                          : %5.3fms, %2d%%\n",
                  fAvgFFT,
                  (int) ((fAvgFFT / fAvgTotal) * 100));
    (void) printf("        calls to Accumulate()/accumulation loop   : %5.3fms, %2d%%\n",
                  fAvgAccum,
                  (int) ((fAvgAccum / fAvgTotal) * 100));
    (void) printf("        x4 calls to cudaMemcpy(Device2Host)       : %5.3fms, %2d%%\n",
                  fAvgCpOut,
                  (int) ((fAvgCpOut / fAvgTotal) * 100));
    VEGASCUDASafeCall(cudaEventDestroy(g_cuStart));
    VEGASCUDASafeCall(cudaEventDestroy(g_cuStop));
    (void) printf("* Events destroyed.\n");
    (void) printf("* Benchmarking run completed.\n");
#endif

    return GUPPI_OK;
}

/* function that creates the FFT plan, allocates memory, initialises counters,
   etc. */
int Init()
{
    int i = 0;
    int iDevCount = 0;
    cudaDeviceProp stDevProp = {0};
    int iRet = GUPPI_OK;

    iRet = RegisterSignalHandlers();
    if (iRet != GUPPI_OK)
    {
        (void) fprintf(stderr, "ERROR: Signal-handler registration failed!\n");
        return GUPPI_ERR_GEN;
    }

    VEGASCUDASafeCall(cudaGetDeviceCount(&iDevCount));
    if (0 == iDevCount)
    {
        (void) fprintf(stderr, "ERROR: No CUDA-capable device found!\n");
        return EXIT_FAILURE;
    }
    else if (iDevCount > 1)
    {
        /* TODO: figure this out */
        (void) fprintf(stderr,
                       "ERROR: More than one CUDA-capable device "
                       "found! Don't know how to proceed!\n");
        return EXIT_FAILURE;
    }

    /* TODO: make it automagic */
    VEGASCUDASafeCall(cudaSetDevice(0));

    VEGASCUDASafeCall(cudaGetDeviceProperties(&stDevProp, 0));
    g_iMaxThreadsPerBlock = stDevProp.maxThreadsPerBlock;

    /* create conversion map from (signed, 2's complement) Fix_8_7 format to
     * float:
     * 1000 0000 (-128) = -1.0
     * to
     * 0111 1111 (127) = 0.9921875 */
    for (i = 0; i < 128; ++i)
    {
        g_afConvLookup[i] = ((float) i) / 128;
    }
    for (i = 128; i < 256; ++i)
    {
        g_afConvLookup[i] = -((float) (256 - i)) / 128;
    }

    /* allocate memory for the lookup table on the device and copy contents */
    VEGASCUDASafeCall(cudaMalloc((void **) &g_pfConvLookup_d, 256 * sizeof(float)));
    VEGASCUDASafeCall(cudaMemcpy(g_pfConvLookup_d,
               g_afConvLookup,
               256 * sizeof(float),
               cudaMemcpyHostToDevice));

    if (g_iIsPFBOn)
    {
        /* set number of taps to NUM_TAPS if PFB is on, else number of
           taps = 1 */
        g_iNTaps = NUM_TAPS;

        g_pcPFBCoeff = (signed char *) malloc(g_iNTaps * g_iNFFT * sizeof(signed char));
        if (NULL == g_pcPFBCoeff)
        {
            (void) fprintf(stderr,
                           "ERROR: Memory allocation failed! %s.\n",
                           strerror(errno));
            return GUPPI_ERR_GEN;
        }

        /* allocate memory for the filter coefficient array on the device */
        VEGASCUDASafeCall(cudaMalloc((void **) &g_pcPFBCoeff_d,
                              g_iNTaps * g_iNFFT * sizeof(signed char)));

        /* read filter coefficients */
        /* build file name */
        (void) sprintf(g_acFileCoeff,
                       "%s%d_%d%s",
                       FILE_COEFF_PREFIX,
                       g_iNTaps,
                       g_iNFFT,
                       FILE_COEFF_SUFFIX);
        g_iFileCoeff = open(g_acFileCoeff, O_RDONLY);
        if (GUPPI_ERR_GEN == g_iFileCoeff)
        {
            (void) fprintf(stderr,
                           "ERROR: Opening filter coefficients file %s failed! %s.\n",
                           g_acFileCoeff,
                           strerror(errno));
            return GUPPI_ERR_GEN;
        }

        iRet = read(g_iFileCoeff, g_pcPFBCoeff, g_iNTaps * g_iNFFT * sizeof(signed char));
        if (GUPPI_ERR_GEN == iRet)
        {
            (void) fprintf(stderr,
                           "ERROR: Reading filter coefficients failed! %s.\n",
                           strerror(errno));
            return GUPPI_ERR_GEN;
        }
        (void) close(g_iFileCoeff);

        /* copy filter coefficients to the device */
        VEGASCUDASafeCall(cudaMemcpy(g_pcPFBCoeff_d,
                   g_pcPFBCoeff,
                   g_iNTaps * g_iNFFT * sizeof(signed char),
                   cudaMemcpyHostToDevice));
    }

    /* allocate memory for data array contents */
    VEGASCUDASafeCall(cudaMalloc((void **) &g_pabDataX_d, 2 * g_iNTaps * g_iNFFT * sizeof(signed char)));
    VEGASCUDASafeCall(cudaMemset(g_pabDataX_d, '\0', 2 * g_iNTaps * g_iNFFT * sizeof(signed char)));
    VEGASCUDASafeCall(cudaMalloc((void **) &g_pabDataY_d, 2 * g_iNTaps * g_iNFFT * sizeof(signed char)));
    VEGASCUDASafeCall(cudaMemset(g_pabDataY_d, '\0', 2 * g_iNTaps * g_iNFFT * sizeof(signed char)));
    for (i = 0; i < g_iNTaps; ++i)
    {
        /* memory allocations on the device */
        VEGASCUDASafeCall(cudaMalloc((void **) &g_apbData_d[i], LEN_DATA * sizeof(BYTE)));
    }

    /* temporarily read a file, instead of input buffer */
    g_iFileData = open(g_acFileData, O_RDONLY);
    if (GUPPI_ERR_GEN == g_iFileData)
    {
        (void) fprintf(stderr,
                       "ERROR! Opening data file %s failed! %s.\n",
                       g_acFileData,
                       strerror(errno));
        return GUPPI_ERR_GEN;
    }

    /* load data into memory */
    iRet = LoadData();
    if (iRet != GUPPI_OK)
    {
        (void) fprintf(stderr,
                       "ERROR! Data loading failed!\n");
        return GUPPI_ERR_GEN;
    }

    /* calculate kernel parameters */
    if (LEN_DATA < g_iMaxThreadsPerBlock)
    {
        /* check num block limit */
        g_dimBlockConv.x = LEN_DATA;
    }
    else
    {
        /* check num block limit */
        g_dimBlockConv.x = g_iMaxThreadsPerBlock;
    }
    g_dimGridConv.x = (int) ceilf(((float) LEN_DATA) / g_iMaxThreadsPerBlock);
    if (g_iNFFT < g_iMaxThreadsPerBlock)
    {
        g_dimBlockUnpack.x = g_iNFFT;
        g_dimBlockPFB.x = g_iNFFT;
        g_dimBlockCopy.x = g_iNFFT;
        g_dimBlockAccum.x = g_iNFFT;
    }
    else
    {
        g_dimBlockUnpack.x = g_iMaxThreadsPerBlock;
        g_dimBlockPFB.x = g_iMaxThreadsPerBlock;
        g_dimBlockCopy.x = g_iMaxThreadsPerBlock;
        g_dimBlockAccum.x = g_iMaxThreadsPerBlock;
    }
    g_dimGridUnpack.x = (int) ceilf(((float) g_iNFFT) / g_iMaxThreadsPerBlock);
    g_dimGridPFB.x = (int) ceilf(((float) g_iNFFT) / g_iMaxThreadsPerBlock);
    g_dimGridCopy.x = (int) ceilf(((float) g_iNFFT) / g_iMaxThreadsPerBlock);
    g_dimGridAccum.x = (int) ceilf(((float) g_iNFFT) / g_iMaxThreadsPerBlock);

    for (i = 0; i < g_iNTaps; ++i)
    {
        g_pbInBufRead = g_pbInBuf + (i * LEN_DATA);
        VEGASCUDASafeCall(cudaMemcpy(g_apbData_d[i],
                          g_pbInBufRead,
                          LEN_DATA * sizeof(BYTE),
                          cudaMemcpyHostToDevice));
        ++g_iReadCount;
        if (g_iReadCount == g_iNumReads)
        {
            (void) printf("Data read done!\n");
            g_iIsDone = TRUE;
        }

        Unpack<<<g_dimGridUnpack, g_dimBlockUnpack>>>(((signed char *) g_pabDataX_d) + (2 * i * g_iNFFT),
                                                      ((signed char *) g_pabDataY_d) + (2 * i * g_iNFFT),
                                                      g_apbData_d[i]);
        VEGASCUDASafeCall(cudaThreadSynchronize());
    }

    g_iPFBWriteIdx = 0;     /* next write into the first buffer */
    g_iPFBReadIdx = 0;      /* PFB to be performed from first buffer */

    g_pccFFTInX = (cufftComplex *) malloc(g_iNFFT * sizeof(cufftComplex));
    if (NULL == g_pccFFTInX)
    {
        (void) fprintf(stderr,
                       "ERROR: Memory allocation failed! %s.\n",
                       strerror(errno));
        return GUPPI_ERR_GEN;
    }
    VEGASCUDASafeCall(cudaMalloc((void **) &g_pccFFTInX_d,
                          g_iNFFT * sizeof(cufftComplex)));
    g_pccFFTInY = (cufftComplex *) malloc(g_iNFFT * sizeof(cufftComplex));
    if (NULL == g_pccFFTInY)
    {
        (void) fprintf(stderr,
                       "ERROR: Memory allocation failed! %s.\n",
                       strerror(errno));
        return GUPPI_ERR_GEN;
    }
    VEGASCUDASafeCall(cudaMalloc((void **) &g_pccFFTInY_d,
                      g_iNFFT * sizeof(cufftComplex)));
    g_pccFFTOutX = (cufftComplex *) malloc(g_iNFFT * sizeof(cufftComplex));
    if (NULL == g_pccFFTOutX)
    {
        (void) fprintf(stderr,
                       "ERROR: Memory allocation failed! %s.\n",
                       strerror(errno));
        return GUPPI_ERR_GEN;
    }
    VEGASCUDASafeCall(cudaMalloc((void **) &g_pccFFTOutX_d,
                      g_iNFFT * sizeof(cufftComplex)));
    g_pccFFTOutY = (cufftComplex *) malloc(g_iNFFT * sizeof(cufftComplex));
    if (NULL == g_pccFFTOutY)
    {
        (void) fprintf(stderr,
                       "ERROR: Memory allocation failed! %s.\n",
                       strerror(errno));
        return GUPPI_ERR_GEN;
    }
    VEGASCUDASafeCall(cudaMalloc((void **) &g_pccFFTOutY_d,
                      g_iNFFT * sizeof(cufftComplex)));

    g_pfSumPowX = (float *) calloc(g_iNFFT, sizeof(float));
    if (NULL == g_pfSumPowX)
    {
        (void) fprintf(stderr,
                       "ERROR: Memory allocation failed! %s.\n",
                       strerror(errno));
        return GUPPI_ERR_GEN;
    }
    g_pfSumPowY = (float *) calloc(g_iNFFT, sizeof(float));
    if (NULL == g_pfSumPowY)
    {
        (void) fprintf(stderr,
                       "ERROR: Memory allocation failed! %s.\n",
                       strerror(errno));
        return GUPPI_ERR_GEN;
    }
    g_pfSumStokesRe = (float *) calloc(g_iNFFT, sizeof(float));
    if (NULL == g_pfSumStokesRe)
    {
        (void) fprintf(stderr,
                       "ERROR: Memory allocation failed! %s.\n",
                       strerror(errno));
        return GUPPI_ERR_GEN;
    }
    g_pfSumStokesIm = (float *) calloc(g_iNFFT, sizeof(float));
    if (NULL == g_pfSumStokesIm)
    {
        (void) fprintf(stderr,
                       "ERROR: Memory allocation failed! %s.\n",
                       strerror(errno));
        return GUPPI_ERR_GEN;
    }
    VEGASCUDASafeCall(cudaMalloc((void **) &g_pfSumPowX_d, g_iNFFT * sizeof(float)));
    VEGASCUDASafeCall(cudaMemset(g_pfSumPowX_d, '\0', g_iNFFT * sizeof(float)));
    VEGASCUDASafeCall(cudaMalloc((void **) &g_pfSumPowY_d, g_iNFFT * sizeof(float)));
    VEGASCUDASafeCall(cudaMemset(g_pfSumPowY_d, '\0', g_iNFFT * sizeof(float)));
    VEGASCUDASafeCall(cudaMalloc((void **) &g_pfSumStokesRe_d, g_iNFFT * sizeof(float)));
    VEGASCUDASafeCall(cudaMemset(g_pfSumStokesRe_d, '\0', g_iNFFT * sizeof(float)));
    VEGASCUDASafeCall(cudaMalloc((void **) &g_pfSumStokesIm_d, g_iNFFT * sizeof(float)));
    VEGASCUDASafeCall(cudaMemset(g_pfSumStokesIm_d, '\0', g_iNFFT * sizeof(float)));

    /* create plans */
    (void) cufftPlan1d(&g_stPlanX, g_iNFFT, CUFFT_C2C, 1);
    (void) cufftPlan1d(&g_stPlanY, g_iNFFT, CUFFT_C2C, 1);

#if PLOT
    /* just for plotting */
    InitPlot();
#endif

    return GUPPI_OK;
}

/* function that reads data from the data file and loads it into memory during
   initialisation */
int LoadData()
{
    struct stat stFileStats = {0};
    int iRet = GUPPI_OK;

    iRet = stat(g_acFileData, &stFileStats);
    if (iRet != GUPPI_OK)
    {
        (void) fprintf(stderr,
                       "ERROR: Failed to stat %s: %s!\n",
                       g_acFileData,
                       strerror(errno));
        return GUPPI_ERR_GEN;
    }

    g_pbInBuf = (BYTE *) malloc(stFileStats.st_size * sizeof(BYTE));
    if (NULL == g_pbInBuf)
    {
        (void) fprintf(stderr,
                       "ERROR: Memory allocation failed! %s.\n",
                       strerror(errno));
        return GUPPI_ERR_GEN;
    }

    iRet = read(g_iFileData, g_pbInBuf, stFileStats.st_size);
    if (GUPPI_ERR_GEN == iRet)
    {
        (void) fprintf(stderr,
                       "ERROR: Data reading failed! %s.\n",
                       strerror(errno));
        return GUPPI_ERR_GEN;
    }
    else if (iRet != stFileStats.st_size)
    {
        (void) printf("File read done!\n");
    }

    /* calculate the number of reads required */
    g_iNumReads = stFileStats.st_size / LEN_DATA;

    return GUPPI_OK;
}

/* function that reads data from input buffer */
int ReadData()
{
    /* write new data to the write buffer */
    g_pbInBufRead += LEN_DATA;
#if BENCHMARKING
    VEGASCUDASafeCall(cudaEventRecord(g_cuStart, 0));
    VEGASCUDASafeCall(cudaEventSynchronize(g_cuStart));
#endif
    VEGASCUDASafeCall(cudaMemcpy(g_apbData_d[g_iPFBWriteIdx],
                      g_pbInBufRead,
                      LEN_DATA * sizeof(BYTE),
                      cudaMemcpyHostToDevice));
#if BENCHMARKING
    VEGASCUDASafeCall(cudaEventRecord(g_cuStop, 0));
    VEGASCUDASafeCall(cudaEventSynchronize(g_cuStop));
    VEGASCUDASafeCall(cudaEventElapsedTime(&g_fTimeCpIn, g_cuStart, g_cuStop));
    g_fAvgCpIn = (g_fTimeCpIn + ((g_iCount - 1) * g_fAvgCpIn)) / g_iCount;
#endif
    ++g_iReadCount;
    if (g_iReadCount == g_iNumReads)
    {
        (void) printf("Data read done!\n");
        g_iIsDone = TRUE;
    }

#if BENCHMARKING
    VEGASCUDASafeCall(cudaEventRecord(g_cuStart, 0));
    VEGASCUDASafeCall(cudaEventSynchronize(g_cuStart));
#endif
    Unpack<<<g_dimGridUnpack, g_dimBlockUnpack>>>(((signed char *) g_pabDataX_d) + (2 * g_iPFBWriteIdx * g_iNFFT),
                                                  ((signed char *) g_pabDataY_d) + (2 * g_iPFBWriteIdx * g_iNFFT),
                                                  g_apbData_d[g_iPFBWriteIdx]);
    VEGASCUDASafeCall(cudaThreadSynchronize());
#if BENCHMARKING
    VEGASCUDASafeCall(cudaEventRecord(g_cuStop, 0));
    VEGASCUDASafeCall(cudaEventSynchronize(g_cuStop));
    VEGASCUDASafeCall(cudaEventElapsedTime(&g_fTimeUnpack, g_cuStart, g_cuStop));
    g_fAvgUnpack = (g_fTimeUnpack + ((g_iCount - 1) * g_fAvgUnpack)) / g_iCount;
#endif

    if (g_iPFBWriteIdx != (g_iNTaps - 1))
    {
        ++g_iPFBWriteIdx;
    }
    else
    {
        g_iPFBWriteIdx = 0;
    }
    if (g_iPFBReadIdx != (g_iNTaps - 1))
    {
        ++g_iPFBReadIdx;
    }
    else
    {
        g_iPFBReadIdx = 0;
    }

    return GUPPI_OK;
}

/* function that converts Fix_8_7 format data to floating-point */
__global__ void Convert(float *pfConvLookup, BYTE *pbData, float *pfData)
{
    int i = (blockIdx.x * blockDim.x) + threadIdx.x;

    pfData[i] = pfConvLookup[pbData[i]];

    return;
}

/* unpack data */
/* assuming real and imaginary parts are interleaved, and X and Y are
   interleaved, like so:
   reX, imX, reY, imY, ... */
__global__ void Unpack(signed char *pbDataX, signed char *pbDataY, BYTE *pbData)
{
    int i = (blockIdx.x * blockDim.x) + threadIdx.x;
    signed char (*pabDataX)[][2*NUM_SUBBANDS] = (signed char(*) [][2*NUM_SUBBANDS]) pbDataX;
    signed char (*pabDataY)[][2*NUM_SUBBANDS] = (signed char(*) [][2*NUM_SUBBANDS]) pbDataY;
    
    /* sub-band 0 */
    (*pabDataX)[i][0] = pbData[i*NUM_BYTES_PER_SAMP];
    (*pabDataX)[i][1] = pbData[(i*NUM_BYTES_PER_SAMP)+1];
    (*pabDataY)[i][0] = pbData[(i*NUM_BYTES_PER_SAMP)+2];
    (*pabDataY)[i][1] = pbData[(i*NUM_BYTES_PER_SAMP)+3];
    /* sub-band 1 */
    (*pabDataX)[i][2] = pbData[(i*NUM_BYTES_PER_SAMP)+4];
    (*pabDataX)[i][3] = pbData[(i*NUM_BYTES_PER_SAMP)+5];
    (*pabDataY)[i][2] = pbData[(i*NUM_BYTES_PER_SAMP)+6];
    (*pabDataY)[i][3] = pbData[(i*NUM_BYTES_PER_SAMP)+7];
    /* sub-band 2 */
    (*pabDataX)[i][4] = pbData[(i*NUM_BYTES_PER_SAMP)+8];
    (*pabDataX)[i][5] = pbData[(i*NUM_BYTES_PER_SAMP)+9];
    (*pabDataY)[i][4] = pbData[(i*NUM_BYTES_PER_SAMP)+10];
    (*pabDataY)[i][5] = pbData[(i*NUM_BYTES_PER_SAMP)+11];
    /* sub-band 3 */
    (*pabDataX)[i][6] = pbData[(i*NUM_BYTES_PER_SAMP)+12];
    (*pabDataX)[i][7] = pbData[(i*NUM_BYTES_PER_SAMP)+13];
    (*pabDataY)[i][6] = pbData[(i*NUM_BYTES_PER_SAMP)+14];
    (*pabDataY)[i][7] = pbData[(i*NUM_BYTES_PER_SAMP)+15];
    /* sub-band 4 */
    (*pabDataX)[i][8] = pbData[(i*NUM_BYTES_PER_SAMP)+16];
    (*pabDataX)[i][9] = pbData[(i*NUM_BYTES_PER_SAMP)+17];
    (*pabDataY)[i][8] = pbData[(i*NUM_BYTES_PER_SAMP)+18];
    (*pabDataY)[i][9] = pbData[(i*NUM_BYTES_PER_SAMP)+19];
    /* sub-band 5 */
    (*pabDataX)[i][10] = pbData[(i*NUM_BYTES_PER_SAMP)+20];
    (*pabDataX)[i][11] = pbData[(i*NUM_BYTES_PER_SAMP)+21];
    (*pabDataY)[i][10] = pbData[(i*NUM_BYTES_PER_SAMP)+22];
    (*pabDataY)[i][11] = pbData[(i*NUM_BYTES_PER_SAMP)+23];
    /* sub-band 6 */
    (*pabDataX)[i][12] = pbData[(i*NUM_BYTES_PER_SAMP)+24];
    (*pabDataX)[i][13] = pbData[(i*NUM_BYTES_PER_SAMP)+25];
    (*pabDataY)[i][12] = pbData[(i*NUM_BYTES_PER_SAMP)+26];
    (*pabDataY)[i][13] = pbData[(i*NUM_BYTES_PER_SAMP)+27];
    /* sub-band 7 */
    (*pabDataX)[i][14] = pbData[(i*NUM_BYTES_PER_SAMP)+28];
    (*pabDataX)[i][15] = pbData[(i*NUM_BYTES_PER_SAMP)+29];
    (*pabDataY)[i][14] = pbData[(i*NUM_BYTES_PER_SAMP)+30];
    (*pabDataY)[i][15] = pbData[(i*NUM_BYTES_PER_SAMP)+31];

    return;
}

/* function that performs the PFB */
__global__ void DoPFB(signed char *pbDataX,
                      signed char *pbDataY,
                      int iPFBReadIdx,
                      int iNTaps,
                      signed char *pcPFBCoeff,
                      cufftComplex *pccFFTInX_0,
                      cufftComplex *pccFFTInY_0,
                      cufftComplex *pccFFTInX_1,
                      cufftComplex *pccFFTInY_1,
                      cufftComplex *pccFFTInX_2,
                      cufftComplex *pccFFTInY_2,
                      cufftComplex *pccFFTInX_3,
                      cufftComplex *pccFFTInY_3,
                      cufftComplex *pccFFTInX_4,
                      cufftComplex *pccFFTInY_4,
                      cufftComplex *pccFFTInX_5,
                      cufftComplex *pccFFTInY_5,
                      cufftComplex *pccFFTInX_6,
                      cufftComplex *pccFFTInY_6,
                      cufftComplex *pccFFTInX_7,
                      cufftComplex *pccFFTInY_7)
{
    int i = iPFBReadIdx;
    int j = 0;
    int k = (blockIdx.x * blockDim.x) + threadIdx.x;
    int iNFFT = gridDim.x * blockDim.x;
    cufftComplex ccAccumX[8];
    cufftComplex ccAccumY[8];
    signed char (*pabDataX)[][2*NUM_SUBBANDS] = (signed char(*) [][2*NUM_SUBBANDS]) pbDataX;
    signed char (*pabDataY)[][2*NUM_SUBBANDS] = (signed char(*) [][2*NUM_SUBBANDS]) pbDataY;
    signed char cPFBCoeff = 0;

    ccAccumX.x = 0.0;
    ccAccumX.y = 0.0;
    ccAccumY.x = 0.0;
    ccAccumY.y = 0.0;

    for (j = 0; j < iNTaps; ++j)
    {
        cPFBCoeff = pcPFBCoeff[(j * iNFFT) + k];
        /* sub-band 0 */
        ccAccumX[0].x += (*pabDataX)[(i * iNFFT) + k][0] * cPFBCoeff;
        ccAccumX[0].y += (*pabDataX)[(i * iNFFT) + k][1] * cPFBCoeff;
        ccAccumY[0].x += (*pabDataY)[(i * iNFFT) + k][0] * cPFBCoeff;
        ccAccumY[0].y += (*pabDataY)[(i * iNFFT) + k][1] * cPFBCoeff;
        /* sub-band 1 */
        ccAccumX[1].x += (*pabDataX)[(i * iNFFT) + k][2] * cPFBCoeff;
        ccAccumX[1].y += (*pabDataX)[(i * iNFFT) + k][3] * cPFBCoeff;
        ccAccumY[1].x += (*pabDataY)[(i * iNFFT) + k][2] * cPFBCoeff;
        ccAccumY[1].y += (*pabDataY)[(i * iNFFT) + k][3] * cPFBCoeff;
        /* sub-band 2 */
        ccAccumX[2].x += (*pabDataX)[(i * iNFFT) + k][4] * cPFBCoeff;
        ccAccumX[2].y += (*pabDataX)[(i * iNFFT) + k][5] * cPFBCoeff;
        ccAccumY[2].x += (*pabDataY)[(i * iNFFT) + k][4] * cPFBCoeff;
        ccAccumY[2].y += (*pabDataY)[(i * iNFFT) + k][5] * cPFBCoeff;
        /* sub-band 3 */
        ccAccumX[3].x += (*pabDataX)[(i * iNFFT) + k][6] * cPFBCoeff;
        ccAccumX[3].y += (*pabDataX)[(i * iNFFT) + k][7] * cPFBCoeff;
        ccAccumY[3].x += (*pabDataY)[(i * iNFFT) + k][6] * cPFBCoeff;
        ccAccumY[3].y += (*pabDataY)[(i * iNFFT) + k][7] * cPFBCoeff;
        /* sub-band 4 */
        ccAccumX[4].x += (*pabDataX)[(i * iNFFT) + k][8] * cPFBCoeff;
        ccAccumX[4].y += (*pabDataX)[(i * iNFFT) + k][9] * cPFBCoeff;
        ccAccumY[4].x += (*pabDataY)[(i * iNFFT) + k][8] * cPFBCoeff;
        ccAccumY[4].y += (*pabDataY)[(i * iNFFT) + k][9] * cPFBCoeff;
        /* sub-band 5 */
        ccAccumX[5].x += (*pabDataX)[(i * iNFFT) + k][10] * cPFBCoeff;
        ccAccumX[5].y += (*pabDataX)[(i * iNFFT) + k][11] * cPFBCoeff;
        ccAccumY[5].x += (*pabDataY)[(i * iNFFT) + k][10] * cPFBCoeff;
        ccAccumY[5].y += (*pabDataY)[(i * iNFFT) + k][11] * cPFBCoeff;
        /* sub-band 6 */
        ccAccumX[6].x += (*pabDataX)[(i * iNFFT) + k][12] * cPFBCoeff;
        ccAccumX[6].y += (*pabDataX)[(i * iNFFT) + k][13] * cPFBCoeff;
        ccAccumY[6].x += (*pabDataY)[(i * iNFFT) + k][12] * cPFBCoeff;
        ccAccumY[6].y += (*pabDataY)[(i * iNFFT) + k][13] * cPFBCoeff;
        /* sub-band 7 */
        ccAccumX[7].x += (*pabDataX)[(i * iNFFT) + k][14] * cPFBCoeff;
        ccAccumX[7].y += (*pabDataX)[(i * iNFFT) + k][15] * cPFBCoeff;
        ccAccumY[7].x += (*pabDataY)[(i * iNFFT) + k][14] * cPFBCoeff;
        ccAccumY[7].y += (*pabDataY)[(i * iNFFT) + k][15] * cPFBCoeff;
        if (i != (iNTaps - 1))
        {
            ++i;
        }
        else
        {
            i = 0;
        }
    }

    pccFFTInX_0[k] = ccAccumX[0];
    pccFFTInY_0[k] = ccAccumY[0];
    pccFFTInX_1[k] = ccAccumX[1];
    pccFFTInY_1[k] = ccAccumY[1];
    pccFFTInX_2[k] = ccAccumX[2];
    pccFFTInY_2[k] = ccAccumY[2];
    pccFFTInX_3[k] = ccAccumX[3];
    pccFFTInY_3[k] = ccAccumY[3];
    pccFFTInX_4[k] = ccAccumX[4];
    pccFFTInY_4[k] = ccAccumY[4];
    pccFFTInX_5[k] = ccAccumX[5];
    pccFFTInY_5[k] = ccAccumY[5];
    pccFFTInX_6[k] = ccAccumX[6];
    pccFFTInY_6[k] = ccAccumY[6];
    pccFFTInX_7[k] = ccAccumX[7];
    pccFFTInY_7[k] = ccAccumY[7];

    return;
}

__global__ void CopyDataForFFT(cufftComplex *pccFFTInX,
                               cufftComplex *pccFFTInY,
                               signed char *pbDataX,
                               signed char *pbDataY,
                               int iNFFT,
                               float *pfConvLookup)
{
    int i = (blockIdx.x * blockDim.x) + threadIdx.x;
    signed char (*pabDataX)[][2] = (signed char(*) [][2]) pbDataX;
    signed char (*pabDataY)[][2] = (signed char(*) [][2]) pbDataY;

    pccFFTInX[i].x = (float) (*pabDataX)[i][0];
    pccFFTInX[i].y = (float) (*pabDataX)[i][1];
    pccFFTInY[i].x = (float) (*pabDataY)[i][0];
    pccFFTInY[i].y = (float) (*pabDataY)[i][1];

    return;
}

/* function that performs the FFT */
int DoFFT()
{
    /* execute plan */
    (void) cufftExecC2C(g_stPlanX, g_pccFFTInX_d, g_pccFFTOutX_d, CUFFT_FORWARD);
    (void) cufftExecC2C(g_stPlanY, g_pccFFTInY_d, g_pccFFTOutY_d, CUFFT_FORWARD);

    return GUPPI_OK;
}

__global__ void Accumulate(cufftComplex *pccFFTOutX,
                           cufftComplex *pccFFTOutY,
                           float *pfSumPowX,
                           float *pfSumPowY,
                           float *pfSumStokesRe,
                           float *pfSumStokesIm)
{
    int i = (blockIdx.x * blockDim.x) + threadIdx.x;

    /* Re(X)^2 + Im(X)^2 */
    pfSumPowX[i] += (pccFFTOutX[i].x * pccFFTOutX[i].x)
                    + (pccFFTOutX[i].y * pccFFTOutX[i].y);
    /* Re(Y)^2 + Im(Y)^2 */
    pfSumPowY[i] += (pccFFTOutY[i].x * pccFFTOutY[i].x)
                    + (pccFFTOutY[i].y * pccFFTOutY[i].y);
    /* Re(XY*) */
    pfSumStokesRe[i] += (pccFFTOutX[i].x * pccFFTOutY[i].x)
                        + (pccFFTOutX[i].y * pccFFTOutY[i].y);
    /* Im(XY*) */
    pfSumStokesIm[i] += (pccFFTOutX[i].y * pccFFTOutY[i].x)
                        - (pccFFTOutX[i].x * pccFFTOutY[i].y);

    return;
}

int IsRunning()
{
    return (!g_iIsDone);
}

int IsBlankingSet()
{
    /* check for status and return TRUE or FALSE */
    return FALSE;
}

/* function that frees resources */
void CleanUp()
{
    int i = 0;

    /* free resources */
    (void) cudaFree(g_pfConvLookup_d);

    free(g_pbInBuf);

    for (i = 0; i < g_iNTaps; ++i)
    {
        (void) cudaFree(g_apbData_d[i]);
    }
    (void) cudaFree(g_pabDataX_d);
    (void) cudaFree(g_pabDataY_d);
    free(g_pccFFTInX);
    (void) cudaFree(g_pccFFTInX_d);
    free(g_pccFFTInY);
    (void) cudaFree(g_pccFFTInY_d);
    free(g_pccFFTOutX);
    (void) cudaFree(g_pccFFTOutX_d);
    free(g_pccFFTOutY);
    (void) cudaFree(g_pccFFTOutY_d);

    free(g_pcPFBCoeff);
    (void) cudaFree(g_pcPFBCoeff_d);

    free(g_pfSumPowX);
    free(g_pfSumPowY);
    free(g_pfSumStokesRe);
    free(g_pfSumStokesIm);

    /* destroy plans */
    (void) cufftDestroy(g_stPlanX);
    (void) cufftDestroy(g_stPlanY);

    (void) close(g_iFileData);

#if PLOT
    /* for plotting */
    free(g_pfFreq);
    cpgclos();
#endif

    return;
}

#if PLOT
void InitPlot()
{
    int iRet = GUPPI_OK;
    int i = 0;

    iRet = cpgopen(PG_DEV);
    if (iRet <= 0)
    {
        (void) fprintf(stderr,
                       "ERROR: Opening graphics device %s failed!\n",
                       PG_DEV);
        return;
    }

    cpgsch(2);
    cpgsubp(1, 4);

    g_pfFreq = (float *) malloc(g_iNFFT * sizeof(float));
    if (NULL == g_pfFreq)
    {
        (void) fprintf(stderr,
                       "ERROR: Memory allocation failed! %s.\n",
                       strerror(errno));
        return;
    }

    /* load the frequency axis */
    for (i = 0; i < g_iNFFT; ++i)
    {
        g_pfFreq[i] = ((float) i * g_fFSamp) / g_iNFFT;
    }

    return;
}

void Plot()
{
    float fMinFreq = g_pfFreq[0];
    float fMaxFreq = g_pfFreq[g_iNFFT-1];
    float fMinY = FLT_MAX;
    float fMaxY = -(FLT_MAX);
    int i = 0;

    /* take log10 of data */
    for (i = 0; i < g_iNFFT; ++i)
    {
        g_pfSumPowX[i] = 10 * log10f(g_pfSumPowX[i]);
        g_pfSumPowY[i] = 10 * log10f(g_pfSumPowY[i]);
        g_pfSumStokesRe[i] = log10f(g_pfSumStokesRe[i]);
        g_pfSumStokesIm[i] = log10f(g_pfSumStokesIm[i]);
    }

    /* plot g_pfSumPowX */
    for (i = 0; i < g_iNFFT; ++i)
    {
        if (g_pfSumPowX[i] > fMaxY)
        {
            fMaxY = g_pfSumPowX[i];
        }
        if (g_pfSumPowX[i] < fMinY)
        {
            fMinY = g_pfSumPowX[i];
        }
    }
    cpgpanl(1, 1);
    cpgeras();
    cpgsvp(PG_VP_ML, PG_VP_MR, PG_VP_MB, PG_VP_MT);
    cpgswin(fMinFreq, fMaxFreq, fMinY, fMaxY);
    cpglab("Bin Number",
           "",
           "SumPowX");
    cpgbox("BCNST", 0.0, 0, "BCNST", 0.0, 0);
    cpgsci(PG_CI_PLOT);
    cpgline(g_iNFFT, g_pfFreq, g_pfSumPowX);
    cpgsci(PG_CI_DEF);

    /* plot g_pfSumPowY */
    fMinY = FLT_MAX;
    fMaxY = -(FLT_MAX);
    for (i = 0; i < g_iNFFT; ++i)
    {
        if (g_pfSumPowY[i] > fMaxY)
        {
            fMaxY = g_pfSumPowY[i];
        }
        if (g_pfSumPowY[i] < fMinY)
        {
            fMinY = g_pfSumPowY[i];
        }
    }
    for (i = 0; i < g_iNFFT; ++i)
    {
        g_pfSumPowY[i] -= fMaxY;
        //printf("%g\n", g_pfSumPowY[i]);
    }
    fMinY -= fMaxY;
    fMaxY = 0;
    //printf("********************************\n");
    cpgpanl(1, 2);
    cpgeras();
    cpgsvp(PG_VP_ML, PG_VP_MR, PG_VP_MB, PG_VP_MT);
    cpgswin(fMinFreq, fMaxFreq, fMinY, fMaxY);
    cpglab("Bin Number",
           "",
           "SumPowY");
    cpgbox("BCNST", 0.0, 0, "BCNST", 0.0, 0);
    cpgsci(PG_CI_PLOT);
    cpgline(g_iNFFT, g_pfFreq, g_pfSumPowY);
    cpgsci(PG_CI_DEF);

    /* plot g_pfSumStokesRe */
    fMinY = FLT_MAX;
    fMaxY = -(FLT_MAX);
    for (i = 0; i < g_iNFFT; ++i)
    {
        if (g_pfSumStokesRe[i] > fMaxY)
        {
            fMaxY = g_pfSumStokesRe[i];
        }
        if (g_pfSumStokesRe[i] < fMinY)
        {
            fMinY = g_pfSumStokesRe[i];
        }
    }
    cpgpanl(1, 3);
    cpgeras();
    cpgsvp(PG_VP_ML, PG_VP_MR, PG_VP_MB, PG_VP_MT);
    cpgswin(fMinFreq, fMaxFreq, fMinY, fMaxY);
    cpglab("Bin Number",
           "",
           "SumStokesRe");
    cpgbox("BCNST", 0.0, 0, "BCNST", 0.0, 0);
    cpgsci(PG_CI_PLOT);
    cpgline(g_iNFFT, g_pfFreq, g_pfSumStokesRe);
    cpgsci(PG_CI_DEF);

    /* plot g_pfSumStokesIm */
    fMinY = FLT_MAX;
    fMaxY = -(FLT_MAX);
    for (i = 0; i < g_iNFFT; ++i)
    {
        if (g_pfSumStokesIm[i] > fMaxY)
        {
            fMaxY = g_pfSumStokesIm[i];
        }
        if (g_pfSumStokesIm[i] < fMinY)
        {
            fMinY = g_pfSumStokesIm[i];
        }
    }
    cpgpanl(1, 4);
    cpgeras();
    cpgsvp(PG_VP_ML, PG_VP_MR, PG_VP_MB, PG_VP_MT);
    cpgswin(fMinFreq, fMaxFreq, fMinY, fMaxY);
    cpglab("Bin Number",
           "",
           "SumStokesIm");
    cpgbox("BCNST", 0.0, 0, "BCNST", 0.0, 0);
    cpgsci(PG_CI_PLOT);
    cpgline(g_iNFFT, g_pfFreq, g_pfSumStokesIm);
    cpgsci(PG_CI_DEF);

    return;
}
#endif

/*
 * Registers handlers for SIGTERM and CTRL+C
 */
int RegisterSignalHandlers()
{
    struct sigaction stSigHandler = {{0}};
    int iRet = GUPPI_OK;

    /* register the CTRL+C-handling function */
    stSigHandler.sa_handler = HandleStopSignals;
    iRet = sigaction(SIGINT, &stSigHandler, NULL);
    if (iRet != GUPPI_OK)
    {
        (void) fprintf(stderr,
                       "ERROR: Handler registration failed for signal %d!\n",
                       SIGINT);
        return GUPPI_ERR_GEN;
    }

    /* register the SIGTERM-handling function */
    stSigHandler.sa_handler = HandleStopSignals;
    iRet = sigaction(SIGTERM, &stSigHandler, NULL);
    if (iRet != GUPPI_OK)
    {
        (void) fprintf(stderr,
                       "ERROR: Handler registration failed for signal %d!\n",
                       SIGTERM);
        return GUPPI_ERR_GEN;
    }

    return GUPPI_OK;
}

/*
 * Catches SIGTERM and CTRL+C and cleans up before exiting
 */
void HandleStopSignals(int iSigNo)
{
    /* clean up */
    CleanUp();

    /* exit */
    exit(GUPPI_OK);

    /* never reached */
    return;
}

void __VEGASCUDASafeCall(cudaError_t iRet,
                         const char* pcFile,
                         const int iLine,
                         void (*pCleanUp)(void))
{
    if (iRet != cudaSuccess)
    {
        (void) fprintf(stderr,
                       "ERROR: File <%s>, Line %d: %s\n",
                       pcFile,
                       iLine,
                       cudaGetErrorString(iRet));
        /* free resources */
        (*pCleanUp)();
        exit(GUPPI_ERR_GEN);
    }

    return;
}

/*
 * Prints usage information
 */
void PrintUsage(const char *pcProgName)
{
    (void) printf("Usage: %s [options] <data-file>\n",
                  pcProgName);
    (void) printf("    -h  --help                           ");
    (void) printf("Display this usage information\n");
    (void) printf("    -n  --nfft <value>                   ");
    (void) printf("Number of points in FFT\n");
    (void) printf("    -p  --pfb                            ");
    (void) printf("Enable PFB\n");
    (void) printf("    -a  --nacc <value>                   ");
    (void) printf("Number of spectra to add\n");
#if PLOT
    (void) printf("    -s  --fsamp <value>                  ");
    (void) printf("Sampling frequency\n");
#endif

    return;
}

