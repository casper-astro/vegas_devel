/* 
 * vegasp3mgpu.cu
 * VEGAS Priority 3 Mode - Stand-Alone GPU Implementation
 *
 * Created by Jayanth Chennamangalam on 2011.06.13
 */

#include "vegasp3mgpu.h"

/* BUG in initcopy */
#define INITCOPY        0
#define OPT             0
#define PAGELOCK        0
#define UNPACKTEX           0

int g_iIsDone = FALSE;

int g_iMaxThreadsPerBlock = 0;

BYTE *g_pbInBuf = NULL;
#if INITCOPY
BYTE *g_pbInBuf_d = NULL;
BYTE *g_pbInBufRead_d = NULL;
#else
BYTE *g_pbInBufRead = NULL;
#endif
int g_iReadCount = 0;
int g_iNumReads = 0;

BYTE* g_apbData_d[NUM_TAPS] = {NULL};   /* raw data, LEN_DATA long*/
float* g_apfData_d[NUM_TAPS] = {NULL};  /* raw data converted to float, LEN_DATA long */
cufftComplex* g_pccDataX_d = NULL;      /* unpacked pol-X data, g_iNFFT long */
cufftComplex* g_pccDataY_d = NULL;      /* unpacked pol-Y data, g_iNFFT long */

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

#if UNPACKTEX
texture<float, 1, cudaReadModeElementType> g_stTexRef;
cudaChannelFormatDesc g_stChFDesc;
#endif

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

int g_iIsPFBOn = DEF_PFB_ON;
int g_iNTaps = 1;                       /* 1 if no PFB, NUM_TAPS if PFB */
int g_iFileCoeff = 0;
char g_acFileCoeff[256] = {0};
float *g_pfPFBCoeff = NULL;
float *g_pfPFBCoeff_d = NULL;
#if OPT
float *g_pfOptPFBCoeff = NULL;
float *g_pfOptPFBCoeff_d = NULL;
#endif
#if CONSTMEM
__device__ __constant__ float g_afPFBCoeff_d[8*1024];
#endif

float g_afConvLookup[256] = {0};
float *g_pfConvLookup_d = NULL;

int g_iFileData = 0;
char g_acFileData[256] = {0};

/* PGPLOT global */
float *g_pfFreq = NULL;
float g_fFSamp = 1.0;                   /* 1 [frequency] */

int main(int argc, char *argv[])
{
    int iRet = GUPPI_OK;
    int i = 0;
    int iTime = 0;
    int iAcc = DEF_ACC;
    struct timeval stStart = {0};
    struct timeval stStop = {0};
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

    (void) gettimeofday(&stStart, NULL);
    while (IsRunning())
    {
        if (g_iIsPFBOn)
        {
            /* do pfb */
            /* reset memory */
            VEGASCUDASafeCall(cudaMemset(g_pccFFTInX_d,
                              '\0',
                              g_iNFFT * sizeof(cufftComplex)));
            VEGASCUDASafeCall(cudaMemset(g_pccFFTInY_d,
                              '\0',
                              g_iNFFT * sizeof(cufftComplex)));

        #if OPT
            DoPFB<<<g_dimGridPFB, g_dimBlockPFB>>>(g_pccDataX_d,
                                                   g_pccDataY_d,
                                                   g_iPFBReadIdx,
                                                   g_iNTaps,
                                                   g_pfOptPFBCoeff_d,
                                                   g_pccFFTInX_d,
                                                   g_pccFFTInY_d);
        #else
            #if CONSTMEM
            DoPFB<<<g_dimGridPFB, g_dimBlockPFB>>>(g_pccDataX_d,
                                                   g_pccDataY_d,
                                                   g_iPFBReadIdx,
                                                   g_iNTaps,
                                                   g_pccFFTInX_d,
                                                   g_pccFFTInY_d);
            #else
            DoPFB<<<g_dimGridPFB, g_dimBlockPFB>>>(g_pccDataX_d,
                                                   g_pccDataY_d,
                                                   g_iPFBReadIdx,
                                                   g_iNTaps,
                                                   g_pfPFBCoeff_d,
                                                   g_pccFFTInX_d,
                                                   g_pccFFTInY_d);
            #endif
        #endif
        }
        else
        {
            #if (!(COPY))
            (void) CopyDataForFFT();
            #else
            CopyDataForFFT<<<g_dimGridCopy, g_dimBlockCopy>>>(g_pccFFTInX_d,
                                                              g_pccFFTInY_d,
                                                              g_pccDataX_d,
                                                              g_pccDataY_d,
                                                              g_iNFFT);
            #endif
        }

        /* do fft */
        (void) DoFFT();

        VEGASCUDASafeCall(cudaMemcpy(g_pccFFTOutX,
                   g_pccFFTOutX_d,
                   g_iNFFT * sizeof(cufftComplex),
                   cudaMemcpyDeviceToHost));
        VEGASCUDASafeCall(cudaMemcpy(g_pccFFTOutY,
                   g_pccFFTOutY_d,
                   g_iNFFT * sizeof(cufftComplex),
                   cudaMemcpyDeviceToHost));

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
                for (i = 0; i < g_iNFFT; ++i)
                {
                    /* Re(X)^2 + Im(X)^2 */
                    g_pfSumPowX[i] += (g_pccFFTOutX[i].x * g_pccFFTOutX[i].x)
                                      + (g_pccFFTOutX[i].y * g_pccFFTOutX[i].y);
                    /* Re(Y)^2 + Im(Y)^2 */
                    g_pfSumPowY[i] += (g_pccFFTOutY[i].x * g_pccFFTOutY[i].x)
                                      + (g_pccFFTOutY[i].y * g_pccFFTOutY[i].y);
                    /* Re(XY*) */
                    g_pfSumStokesRe[i] += (g_pccFFTOutX[i].x * g_pccFFTOutY[i].x)
                                          + (g_pccFFTOutX[i].y * g_pccFFTOutY[i].y);
                    /* Im(XY*) */
                    g_pfSumStokesIm[i] += (g_pccFFTOutX[i].y * g_pccFFTOutY[i].x)
                                          - (g_pccFFTOutX[i].x * g_pccFFTOutY[i].y);
                }
                ++iTime;
                if (iTime == iAcc)
                {
                    #if PLOT
                    /* NOTE: Plot() will modify data! */
                    Plot();
                    usleep(500000);
                    #endif

                    /* dump to buffer */

                    /* reset time */
                    iTime = 0;
                    /* zero accumulators */
                    (void) memset(g_pfSumPowX, '\0', g_iNFFT * sizeof(float));
                    (void) memset(g_pfSumPowY, '\0', g_iNFFT * sizeof(float));
                    (void) memset(g_pfSumStokesRe, '\0', g_iNFFT * sizeof(float));
                    (void) memset(g_pfSumStokesIm, '\0', g_iNFFT * sizeof(float));
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
    (void) gettimeofday(&stStop, NULL);
    (void) printf("Time taken (barring Init()): %gs\n",
                  ((stStop.tv_sec + (stStop.tv_usec * USEC2SEC))
                   - (stStart.tv_sec + (stStart.tv_usec * USEC2SEC))));

    CleanUp();

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

        g_pfPFBCoeff = (float *) malloc(g_iNTaps * g_iNFFT * sizeof(float));
        if (NULL == g_pfPFBCoeff)
        {
            (void) fprintf(stderr,
                           "ERROR: Memory allocation failed! %s.\n",
                           strerror(errno));
            return GUPPI_ERR_GEN;
        }

        /* allocate memory for the filter coefficient array on the device */
        VEGASCUDASafeCall(cudaMalloc((void **) &g_pfPFBCoeff_d,
                              g_iNTaps * g_iNFFT * sizeof(float)));

#if OPT
        g_pfOptPFBCoeff = (float *) malloc(2 * g_iNTaps * g_iNFFT * sizeof(float)); 
        if (NULL == g_pfOptPFBCoeff)
        {
            (void) fprintf(stderr,
                           "ERROR: Memory allocation failed! %s.\n",
                           strerror(errno));
            return GUPPI_ERR_GEN;
        }

        VEGASCUDASafeCall(cudaMalloc((void **) &g_pfOptPFBCoeff_d,
                              2 * g_iNTaps * g_iNFFT * sizeof(float)));
#endif

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

        iRet = read(g_iFileCoeff, g_pfPFBCoeff, g_iNTaps * g_iNFFT * sizeof(float));
        if (GUPPI_ERR_GEN == iRet)
        {
            (void) fprintf(stderr,
                           "ERROR: Reading filter coefficients failed! %s.\n",
                           strerror(errno));
            return GUPPI_ERR_GEN;
        }
        (void) close(g_iFileCoeff);

        /* copy filter coefficients to the device */
        VEGASCUDASafeCall(cudaMemcpy(g_pfPFBCoeff_d,
                   g_pfPFBCoeff,
                   g_iNTaps * g_iNFFT * sizeof(float),
                   cudaMemcpyHostToDevice));

#if CONSTMEM
        VEGASCUDASafeCall(cudaMemcpyToSymbol(g_afPFBCoeff_d,
                           g_pfPFBCoeff,
                           g_iNTaps * g_iNFFT * sizeof(float),
                           0,
                           cudaMemcpyHostToDevice));
#endif

#if OPT
    	/* duplicate the coefficients for PFB optimisation */
        for (i = 0; i < (g_iNTaps * g_iNFFT); ++i)
        {
            g_pfOptPFBCoeff[2*i] = g_pfPFBCoeff[i];
            g_pfOptPFBCoeff[(2*i)+1] = g_pfPFBCoeff[i];
    	}

        VEGASCUDASafeCall(cudaMemcpy(g_pfOptPFBCoeff_d,
                   g_pfOptPFBCoeff,
                   2 * g_iNTaps * g_iNFFT * sizeof(float),
                   cudaMemcpyHostToDevice));
#endif
    }

    /* allocate memory for data array contents */
    VEGASCUDASafeCall(cudaMalloc((void **) &g_pccDataX_d, g_iNTaps * g_iNFFT * sizeof(cufftComplex)));
    VEGASCUDASafeCall(cudaMemset(g_pccDataX_d, '\0', g_iNTaps * g_iNFFT * sizeof(cufftComplex)));
    VEGASCUDASafeCall(cudaMalloc((void **) &g_pccDataY_d, g_iNTaps * g_iNFFT * sizeof(cufftComplex)));
    VEGASCUDASafeCall(cudaMemset(g_pccDataY_d, '\0', g_iNTaps * g_iNFFT * sizeof(cufftComplex)));
    for (i = 0; i < g_iNTaps; ++i)
    {
        /* memory allocations on the device */
        VEGASCUDASafeCall(cudaMalloc((void **) &g_apbData_d[i], LEN_DATA * sizeof(BYTE)));
        VEGASCUDASafeCall(cudaMalloc((void **) &g_apfData_d[i], LEN_DATA * sizeof(float)));
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
    #if OPT
    if ((2 * g_iNFFT) < g_iMaxThreadsPerBlock)
    #else
    if (g_iNFFT < g_iMaxThreadsPerBlock)
    #endif
    {
        g_dimBlockUnpack.x = g_iNFFT;
        #if OPT
        g_dimBlockPFB.x = 2 * g_iNFFT;
        g_dimBlockCopy.x = 2 * g_iNFFT;
        #else
        g_dimBlockPFB.x = g_iNFFT;
        g_dimBlockCopy.x = g_iNFFT;
        #endif
    }
    else
    {
        g_dimBlockUnpack.x = g_iMaxThreadsPerBlock;
        g_dimBlockPFB.x = g_iMaxThreadsPerBlock;
        g_dimBlockCopy.x = g_iMaxThreadsPerBlock;
    }
    g_dimGridUnpack.x = (int) ceilf(((float) g_iNFFT) / g_iMaxThreadsPerBlock);
    #if OPT
    g_dimGridPFB.x = (int) ceilf(((float) (2 * g_iNFFT)) / g_iMaxThreadsPerBlock);
    g_dimGridCopy.x = (int) ceilf(((float) (2 * g_iNFFT)) / g_iMaxThreadsPerBlock);
    #else
    g_dimGridPFB.x = (int) ceilf(((float) g_iNFFT) / g_iMaxThreadsPerBlock);
    g_dimGridCopy.x = (int) ceilf(((float) g_iNFFT) / g_iMaxThreadsPerBlock);
    #endif

#if UNPACKTEX
    g_stChFDesc = cudaCreateChannelDesc<float>();
#endif

    for (i = 0; i < g_iNTaps; ++i)
    {
        #if INITCOPY
        g_pbInBufRead_d = g_pbInBuf_d + (i * LEN_DATA);
        #else
        g_pbInBufRead = g_pbInBuf + (i * LEN_DATA);
        VEGASCUDASafeCall(cudaMemcpy(g_apbData_d[i],
                          g_pbInBufRead,
                          LEN_DATA * sizeof(BYTE),
                          cudaMemcpyHostToDevice));
        #endif
        ++g_iReadCount;
        if (g_iReadCount == g_iNumReads)
        {
            (void) printf("Data read done!\n");
            g_iIsDone = TRUE;
        }

        Convert<<<g_dimGridConv, g_dimBlockConv>>>(g_pfConvLookup_d, g_apbData_d[i], g_apfData_d[i]);
        VEGASCUDASafeCall(cudaThreadSynchronize());

#if UNPACKTEX
        /* bind texture to memory */
        VEGASCUDASafeCall(cudaBindTexture(NULL, &g_stTexRef, g_apfData_d[i], &g_stChFDesc, 4 * g_iNFFT * sizeof(float)));
#endif

        Unpack<<<g_dimGridUnpack, g_dimBlockUnpack>>>(g_pccDataX_d + (i * g_iNFFT), g_pccDataY_d + (i * g_iNFFT), g_apfData_d[i]);
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

    #if PAGELOCK
    VEGASCUDASafeCall(cudaMallocHost((void **) &g_pbInBuf, stFileStats.st_size * sizeof(BYTE)));
    #else
    g_pbInBuf = (BYTE *) malloc(stFileStats.st_size * sizeof(BYTE));
    if (NULL == g_pbInBuf)
    {
        (void) fprintf(stderr,
                       "ERROR: Memory allocation failed! %s.\n",
                       strerror(errno));
        return GUPPI_ERR_GEN;
    }
    #endif
    #if INITCOPY
    VEGASCUDASafeCall(cudaMalloc((void **) &g_pbInBuf_d, stFileStats.st_size * sizeof(BYTE)));
    VEGASCUDASafeCall(cudaMemcpy(g_pbInBuf_d,
               g_pbInBuf,
               stFileStats.st_size * sizeof(BYTE),
               cudaMemcpyHostToDevice));
    #endif

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
    #if INITCOPY
    g_pbInBufRead_d += LEN_DATA;
    #else
    g_pbInBufRead += LEN_DATA;
    VEGASCUDASafeCall(cudaMemcpy(g_apbData_d[g_iPFBWriteIdx],
               g_pbInBufRead,
               LEN_DATA * sizeof(BYTE),
               cudaMemcpyHostToDevice));
    #endif
    ++g_iReadCount;
    if (g_iReadCount == g_iNumReads)
    {
        (void) printf("Data read done!\n");
        g_iIsDone = TRUE;
    }

    /* convert data format */
    Convert<<<g_dimGridConv, g_dimBlockConv>>>(g_pfConvLookup_d,
                                               g_apbData_d[g_iPFBWriteIdx],
                                               g_apfData_d[g_iPFBWriteIdx]);

#if UNPACKTEX
    /* bind texture to memory */
    VEGASCUDASafeCall(cudaBindTexture(NULL, &g_stTexRef, g_apfData_d[g_iPFBWriteIdx], &g_stChFDesc, 4 * g_iNFFT * sizeof(float)));
#endif

    Unpack<<<g_dimGridUnpack, g_dimBlockUnpack>>>(g_pccDataX_d + (g_iPFBWriteIdx * g_iNFFT),
                                                  g_pccDataY_d + (g_iPFBWriteIdx * g_iNFFT),
                                                  g_apfData_d[g_iPFBWriteIdx]);

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
__global__ void Unpack(cufftComplex *pccDataX, cufftComplex *pccDataY, float *pfData)
{
    int i = (blockIdx.x * blockDim.x) + threadIdx.x;
    //int j = i * NUM_BYTES_PER_SAMP; // doesn't really speed up things

#if UNPACKTEX
    pccDataX[i].x = tex1Dfetch(g_stTexRef, (i * NUM_BYTES_PER_SAMP));
    pccDataX[i].y = tex1Dfetch(g_stTexRef, (i * NUM_BYTES_PER_SAMP) + 1);
    pccDataY[i].x = tex1Dfetch(g_stTexRef, (i * NUM_BYTES_PER_SAMP) + 2);
    pccDataY[i].y = tex1Dfetch(g_stTexRef, (i * NUM_BYTES_PER_SAMP) + 3);
#else
    pccDataX[i].x = pfData[i*NUM_BYTES_PER_SAMP];
    pccDataX[i].y = pfData[(i*NUM_BYTES_PER_SAMP)+1];
    pccDataY[i].x = pfData[(i*NUM_BYTES_PER_SAMP)+2];
    pccDataY[i].y = pfData[(i*NUM_BYTES_PER_SAMP)+3];
#endif

    return;
}

/* function that performs the PFB */
#if (!(OPT))
#if CONSTMEM
__global__ void DoPFB(cufftComplex *pccDataX,
                      cufftComplex *pccDataY,
                      int iPFBReadIdx,
                      int iNTaps,
                      cufftComplex *pccFFTInX,
                      cufftComplex *pccFFTInY)
{
    int i = iPFBReadIdx;
    int j = 0;
    int k = (blockIdx.x * blockDim.x) + threadIdx.x;
    int iNFFT = gridDim.x * blockDim.x;
    cufftComplex ccAccumX;
    cufftComplex ccAccumY;
    #if 0
    float fCoeff = 0.0;
    cufftComplex ccDataX;
    cufftComplex ccDataY;
    int iDataIdx = 0;
    int iCoeffIdx = 0;
    #endif

    ccAccumX.x = 0.0;
    ccAccumX.y = 0.0;
    ccAccumY.x = 0.0;
    ccAccumY.y = 0.0;

    for (j = 0; j < iNTaps; ++j)
    {
    #if 1
        ccAccumX.x += pccDataX[(i * iNFFT) + k].x * g_afPFBCoeff_d[(j * iNFFT) + k];
        ccAccumX.y += pccDataX[(i * iNFFT) + k].y * g_afPFBCoeff_d[(j * iNFFT) + k];
        ccAccumY.x += pccDataY[(i * iNFFT) + k].x * g_afPFBCoeff_d[(j * iNFFT) + k];
        ccAccumY.y += pccDataY[(i * iNFFT) + k].y * g_afPFBCoeff_d[(j * iNFFT) + k];
    #endif
    #if 0
        iDataIdx = (i * iNFFT) + k;
        iCoeffIdx = (j * iNFFT) + k;
        ccDataX = pccDataX[iDataIdx];
        ccDataY = pccDataY[iDataIdx];
        fCoeff = pfPFBCoeff[iCoeffIdx];
        ccAccumX.x += ccDataX.x * fCoeff;
        ccAccumX.y += ccDataX.y * fCoeff;
        ccAccumY.x += ccDataY.x * fCoeff;
        ccAccumY.y += ccDataY.y * fCoeff;
    #endif
        if (i != (iNTaps - 1))
        {
            ++i;
        }
        else
        {
            i = 0;
        }
    }

    pccFFTInX[k] = ccAccumX;
    pccFFTInY[k] = ccAccumY;

    return;
}
#else
__global__ void DoPFB(cufftComplex *pccDataX,
                      cufftComplex *pccDataY,
                      int iPFBReadIdx,
                      int iNTaps,
                      float *pfPFBCoeff,
                      cufftComplex *pccFFTInX,
                      cufftComplex *pccFFTInY)
{
    int i = iPFBReadIdx;
    int j = 0;
    int k = (blockIdx.x * blockDim.x) + threadIdx.x;
    int iNFFT = gridDim.x * blockDim.x;
    cufftComplex ccAccumX;
    cufftComplex ccAccumY;
    #if 0
    float fCoeff = 0.0;
    cufftComplex ccDataX;
    cufftComplex ccDataY;
    int iDataIdx = 0;
    int iCoeffIdx = 0;
    #endif

    ccAccumX.x = 0.0;
    ccAccumX.y = 0.0;
    ccAccumY.x = 0.0;
    ccAccumY.y = 0.0;

    for (j = 0; j < iNTaps; ++j)
    {
    #if 1
        ccAccumX.x += pccDataX[(i * iNFFT) + k].x * pfPFBCoeff[(j * iNFFT) + k];
        ccAccumX.y += pccDataX[(i * iNFFT) + k].y * pfPFBCoeff[(j * iNFFT) + k];
        ccAccumY.x += pccDataY[(i * iNFFT) + k].x * pfPFBCoeff[(j * iNFFT) + k];
        ccAccumY.y += pccDataY[(i * iNFFT) + k].y * pfPFBCoeff[(j * iNFFT) + k];
    #endif
    #if 0
        iDataIdx = (i * iNFFT) + k;
        iCoeffIdx = (j * iNFFT) + k;
        ccDataX = pccDataX[iDataIdx];
        ccDataY = pccDataY[iDataIdx];
        fCoeff = pfPFBCoeff[iCoeffIdx];
        ccAccumX.x += ccDataX.x * fCoeff;
        ccAccumX.y += ccDataX.y * fCoeff;
        ccAccumY.x += ccDataY.x * fCoeff;
        ccAccumY.y += ccDataY.y * fCoeff;
    #endif
        if (i != (iNTaps - 1))
        {
            ++i;
        }
        else
        {
            i = 0;
        }
    }

    pccFFTInX[k] = ccAccumX;
    pccFFTInY[k] = ccAccumY;

    return;
}
#endif
#else
__global__ void DoPFB(cufftComplex *pccDataX,
                      cufftComplex *pccDataY,
                      int iPFBReadIdx,
                      int iNTaps,
                      float *pfOptPFBCoeff,
                      cufftComplex *pccFFTInX,
                      cufftComplex *pccFFTInY)
{
    int i = iPFBReadIdx;
    int j = 0;
    int k = (blockIdx.x * blockDim.x) + threadIdx.x;
    int iNFFT = gridDim.x * blockDim.x;
    cufftComplex ccAccumX;
    cufftComplex ccAccumY;
    int iCoeffStartIdx = 0;
	float *pfIn = NULL;
	float *pfCoeff = NULL;
    float *pfFFTInX = (float *) pccFFTInX;
    float *pfFFTInY = (float *) pccFFTInY;

    ccAccumX.x = 0.0;
    ccAccumX.y = 0.0;
    ccAccumY.x = 0.0;
    ccAccumY.y = 0.0;

    for (j = 0; j < iNTaps; ++j)
    {
        iCoeffStartIdx = /*2 */ j * iNFFT;
		pfIn = (float *) &pccDataX[i*iNFFT];
		pfCoeff = &pfOptPFBCoeff[iCoeffStartIdx];
        if (0 == k%2)
        {
            ccAccumX.x += pfIn[k] * pfCoeff[k];
        }
        else
        {
            ccAccumX.y = pfIn[k];
        }
        #if 0
		pfIn = (float *) &pccDataY[i*iNFFT];
		pfCoeff = &pfOptPFBCoeff[iCoeffStartIdx];
        if (0 == k%2)
        {
            ccAccumY.x += pfIn[k] * pfCoeff[k];
        }
        else
        {
            ccAccumY.y += pfIn[k] * pfCoeff[k];
        }
        #endif
        if (i != (iNTaps - 1))
        {
            ++i;
        }
        else
        {
            i = 0;
        }
    }

    if (0 == k%2)
    {
        pfFFTInX[k] = ccAccumX.x;
    }
    else
    {
        pfFFTInX[k] = ccAccumX.y;
    }

    return;
}
#endif

#if (!(COPY))
int CopyDataForFFT()
{
    VEGASCUDASafeCall(cudaMemcpy(g_pccFFTInX_d,
               &g_pccDataX_d[g_iPFBReadIdx * g_iNFFT],
               g_iNFFT * sizeof(cufftComplex),
               cudaMemcpyDeviceToDevice));
    VEGASCUDASafeCall(cudaMemcpy(g_pccFFTInY_d,
               &g_pccDataY_d[g_iPFBReadIdx * g_iNFFT],
               g_iNFFT * sizeof(cufftComplex),
               cudaMemcpyDeviceToDevice));

    return GUPPI_OK;
}
#else
__global__ void CopyDataForFFT(cufftComplex *pccFFTInX,
                               cufftComplex *pccFFTInY,
                               cufftComplex *pccDataX,
                               cufftComplex *pccDataY,
                               int iNFFT)
{
    int i = (blockIdx.x * blockDim.x) + threadIdx.x;

    pccFFTInX[i] = pccDataX[i];
    pccFFTInY[i] = pccDataY[i];

    return;
}
#endif

/* function that performs the FFT */
int DoFFT()
{
    /* execute plan */
    cufftExecC2C(g_stPlanX, g_pccFFTInX_d, g_pccFFTOutX_d, CUFFT_FORWARD);
    cufftExecC2C(g_stPlanY, g_pccFFTInY_d, g_pccFFTOutY_d, CUFFT_FORWARD);

    return GUPPI_OK;
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

    #if PAGELOCK
    (void) cudaFreeHost(g_pbInBuf);
    #else
    free(g_pbInBuf);
    #endif
    #if INITCOPY
    (void) cudaFree(g_pbInBuf_d);
    #endif

    for (i = 0; i < g_iNTaps; ++i)
    {
        (void) cudaFree(g_apbData_d[i]);
        (void) cudaFree(g_apfData_d[i]);
    }
    (void) cudaFree(g_pccDataX_d);
    (void) cudaFree(g_pccDataY_d);
    free(g_pccFFTInX);
    (void) cudaFree(g_pccFFTInX_d);
    free(g_pccFFTInY);
    (void) cudaFree(g_pccFFTInY_d);
    free(g_pccFFTOutX);
    (void) cudaFree(g_pccFFTOutX_d);
    free(g_pccFFTOutY);
    (void) cudaFree(g_pccFFTOutY_d);

    free(g_pfPFBCoeff);

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

