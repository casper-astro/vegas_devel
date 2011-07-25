/** 
 * @file vegas_gpu_standalone.cu
 * VEGAS GPU Modes - Stand-Alone Implementation
 *
 * @author Jayanth Chennamangalam
 * @date 2011.07.08
 */

#include "vegas_gpu_standalone.h"

int g_iIsDataReadDone = FALSE;
int g_iIsProcDone = FALSE;
int g_iMaxThreadsPerBlock = 0;
char4* g_pc4InBuf = NULL;
char4* g_pc4InBufRead = NULL;
int g_iSizeFile = 0;
int g_iReadCount = 0;
char4* g_pc4Data_d = NULL;              /* raw data starting address */
char4* g_pc4DataRead_d = NULL;          /* raw data read pointer */
int g_iNFFT = DEF_LEN_SPEC;
dim3 g_dimBPFB(1, 1, 1);
dim3 g_dimGPFB(1, 1);
dim3 g_dimBCopy(1, 1, 1);
dim3 g_dimGCopy(1, 1);
dim3 g_dimBAccum(1, 1, 1);
dim3 g_dimGAccum(1, 1);
float4* g_pf4FFTIn_d = NULL;
float4* g_pf4FFTOut_d = NULL;
cufftHandle g_stPlan = {0};
float4* g_pf4SumStokes = NULL;
float4* g_pf4SumStokes_d = NULL;
int g_iIsPFBOn = DEF_PFB_ON;
int g_iNTaps = 1;                       /* 1 if no PFB, NUM_TAPS if PFB */
char g_acFileData[256] = {0};
/* TODO: crash if file size is less than 32MB */
int g_iSizeRead = DEF_SIZE_READ;
int g_iNumSubBands = DEF_NUM_SUBBANDS;
int g_iUsePlanMany = FALSE;

#if PLOT
float* g_pfSumPowX = NULL;
float* g_pfSumPowY = NULL;
float* g_pfSumStokesRe = NULL;
float* g_pfSumStokesIm = NULL;
float* g_pfFreq = NULL;
float g_fFSamp = 1.0;                   /* 1 [frequency] */
#endif

#if BENCHMARKING
float g_fTimeCpIn = 0.0;
float g_fTotCpIn = 0.0;
int g_iCountCpIn = 0;
cudaEvent_t g_cuStart;
cudaEvent_t g_cuStop;
#endif

int main(int argc, char *argv[])
{
    int iRet = EXIT_SUCCESS;
    int iTime = 0;
    int iAcc = DEF_ACC;
    int iProcData = 0;
    cudaError_t iCUDARet = cudaSuccess;
#if BENCHMARKING
    float fTimePFB = 0.0;
    float fTotPFB = 0.0;
    int iCountPFB = 0;
    float fTimeCpInFFT = 0.0;
    float fTotCpInFFT = 0.0;
    int iCountCpInFFT = 0;
    float fTimeFFT = 0.0;
    float fTotFFT = 0.0;
    int iCountFFT = 0;
    float fTimeCpOut = 0.0;
    float fTotCpOut = 0.0;
    int iCountCpOut = 0;
    float fTimeAccum = 0.0;
    float fTotAccum = 0.0;
    int iCountAccum = 0;
#else
    struct timeval stStart = {0};
    struct timeval stStop = {0};
#endif
#if OUTFILE
    int iFileSpec = 0;
#endif
    const char *pcProgName = NULL;
    int iNextOpt = 0;
    /* valid short options */
#if PLOT
    const char* const pcOptsShort = "hb:n:pa:s:";
#else
    const char* const pcOptsShort = "hb:n:pa:";
#endif
    /* valid long options */
    const struct option stOptsLong[] = {
        { "help",           0, NULL, 'h' },
        { "nsub",           1, NULL, 'b' },
        { "nfft",           1, NULL, 'n' },
        { "pfb",            0, NULL, 'p' },
        { "nacc",           1, NULL, 'a' },
#if PLOT
        { "fsamp",          1, NULL, 's' },
#endif
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

            case 'b':   /* -b or --nsub */
                /* set option */
                g_iNumSubBands = (int) atoi(optarg);
                break;

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

#if PLOT
            case 's':   /* -s or --fsamp */
                /* set option */
                g_fFSamp = (float) atof(optarg);
                break;
#endif

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
        return EXIT_FAILURE;
    }

    (void) strncpy(g_acFileData, argv[optind], 256);
    g_acFileData[255] = '\0';

#if BENCHMARKING
    (void) printf("* Benchmarking run commencing...\n");
#endif

    /* initialise */
    iRet = Init();
    if (iRet != EXIT_SUCCESS)
    {
        (void) fprintf(stderr, "ERROR! Init failed!\n");
        CleanUp();
        return EXIT_FAILURE;
    }

#if OUTFILE
    iFileSpec = open("spec.dat",
                 O_CREAT | O_TRUNC | O_WRONLY,
                 S_IRUSR | S_IWUSR | S_IRGRP | S_IROTH);
    if (EXIT_FAILURE == iFileSpec)
    {
        (void) fprintf(stderr, "ERROR! Opening spectrum file failed!\n");
        CleanUp();
        return EXIT_FAILURE;
    }
#endif

#if (!BENCHMARKING)
    (void) gettimeofday(&stStart, NULL);
#endif
    while (IsRunning())
    {
        if (g_iIsPFBOn)
        {
            /* do pfb */
#if BENCHMARKING
            CUDASafeCallWithCleanUp(cudaEventRecord(g_cuStart, 0));
            CUDASafeCallWithCleanUp(cudaEventSynchronize(g_cuStart));
#endif
            DoPFB<<<g_dimGPFB, g_dimBPFB>>>(g_pc4DataRead_d,
                                            g_pf4FFTIn_d);
            CUDASafeCallWithCleanUp(cudaThreadSynchronize());
            iCUDARet = cudaGetLastError();
            if (iCUDARet != cudaSuccess)
            {
                (void) fprintf(stderr,
                               "ERROR: File <%s>, Line %d: %s\n",
                               __FILE__,
                               __LINE__,
                               cudaGetErrorString(iCUDARet));
                /* free resources */
                CleanUp();
                return EXIT_FAILURE;
            }
#if BENCHMARKING
            CUDASafeCallWithCleanUp(cudaEventRecord(g_cuStop, 0));
            CUDASafeCallWithCleanUp(cudaEventSynchronize(g_cuStop));
            CUDASafeCallWithCleanUp(cudaEventElapsedTime(&fTimePFB, g_cuStart, g_cuStop));
            fTotPFB += fTimePFB;
            ++iCountPFB;
#endif
            /* update the data read pointer */
            g_pc4DataRead_d += (g_iNumSubBands * g_iNFFT);
        }
        else
        {
#if BENCHMARKING
            CUDASafeCallWithCleanUp(cudaEventRecord(g_cuStart, 0));
            CUDASafeCallWithCleanUp(cudaEventSynchronize(g_cuStart));
#endif
            CopyDataForFFT<<<g_dimGCopy, g_dimBCopy>>>(g_pc4DataRead_d,
                                                       g_pf4FFTIn_d);
            CUDASafeCallWithCleanUp(cudaThreadSynchronize());
            iCUDARet = cudaGetLastError();
            if (iCUDARet != cudaSuccess)
            {
                (void) fprintf(stderr,
                               "ERROR: File <%s>, Line %d: %s\n",
                               __FILE__,
                               __LINE__,
                               cudaGetErrorString(iCUDARet));
                /* free resources */
                CleanUp();
                return EXIT_FAILURE;
            }
#if BENCHMARKING
            CUDASafeCallWithCleanUp(cudaEventRecord(g_cuStop, 0));
            CUDASafeCallWithCleanUp(cudaEventSynchronize(g_cuStop));
            CUDASafeCallWithCleanUp(cudaEventElapsedTime(&fTimeCpInFFT, g_cuStart, g_cuStop));
            fTotCpInFFT += fTimeCpInFFT;
            ++iCountCpInFFT;
#endif
            /* update the data read pointer */
            g_pc4DataRead_d += (g_iNumSubBands * g_iNFFT);
        }

        /* do fft */
#if BENCHMARKING
        CUDASafeCallWithCleanUp(cudaEventRecord(g_cuStart, 0));
        CUDASafeCallWithCleanUp(cudaEventSynchronize(g_cuStart));
#endif
        iRet = DoFFT();
        if (iRet != EXIT_SUCCESS)
        {
            (void) fprintf(stderr, "ERROR! FFT failed!\n");
#if OUTFILE
            (void) close(iFileSpec);
#endif
            CleanUp();
            return EXIT_FAILURE;
        }
#if BENCHMARKING
        CUDASafeCallWithCleanUp(cudaEventRecord(g_cuStop, 0));
        CUDASafeCallWithCleanUp(cudaEventSynchronize(g_cuStop));
        CUDASafeCallWithCleanUp(cudaEventElapsedTime(&fTimeFFT,
                                               g_cuStart,
                                               g_cuStop));
        fTotFFT += fTimeFFT;
        ++iCountFFT;
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
                CUDASafeCallWithCleanUp(cudaMemset(g_pf4SumStokes_d,
                                             '\0',
                                             (g_iNumSubBands
                                              * g_iNFFT
                                              * sizeof(float4))));
            }
            else
            {
#if BENCHMARKING
                CUDASafeCallWithCleanUp(cudaEventRecord(g_cuStart, 0));
                CUDASafeCallWithCleanUp(cudaEventSynchronize(g_cuStart));
#endif
                Accumulate<<<g_dimGAccum, g_dimBAccum>>>(g_pf4FFTOut_d,
                                                         g_pf4SumStokes_d);
                CUDASafeCallWithCleanUp(cudaThreadSynchronize());
                iCUDARet = cudaGetLastError();
                if (iCUDARet != cudaSuccess)
                {
                    (void) fprintf(stderr,
                                   "ERROR: File <%s>, Line %d: %s\n",
                                   __FILE__,
                                   __LINE__,
                                   cudaGetErrorString(iCUDARet));
                    /* free resources */
                    CleanUp();
                    return EXIT_FAILURE;
                }
#if BENCHMARKING
                CUDASafeCallWithCleanUp(cudaEventRecord(g_cuStop, 0));
                CUDASafeCallWithCleanUp(cudaEventSynchronize(g_cuStop));
                CUDASafeCallWithCleanUp(cudaEventElapsedTime(&fTimeAccum,
                                                       g_cuStart,
                                                       g_cuStop));
                fTotAccum += fTimeAccum;
                ++iCountAccum;
#endif
                ++iTime;
                if (iTime == iAcc)
                {
                    /* dump to buffer */
#if BENCHMARKING
                    CUDASafeCallWithCleanUp(cudaEventRecord(g_cuStart, 0));
                    CUDASafeCallWithCleanUp(cudaEventSynchronize(g_cuStart));
#endif
                    CUDASafeCallWithCleanUp(cudaMemcpy(g_pf4SumStokes,
                                                 g_pf4SumStokes_d,
                                                 (g_iNumSubBands
                                                  * g_iNFFT
                                                  * sizeof(float4)),
                                                 cudaMemcpyDeviceToHost));
#if BENCHMARKING
                    CUDASafeCallWithCleanUp(cudaEventRecord(g_cuStop, 0));
                    CUDASafeCallWithCleanUp(cudaEventSynchronize(g_cuStop));
                    CUDASafeCallWithCleanUp(cudaEventElapsedTime(&fTimeCpOut,
                                                           g_cuStart,
                                                           g_cuStop));
                    fTotCpOut += fTimeCpOut;
                    ++iCountCpOut;
#endif

#if OUTFILE
                    for (i = 0; i < g_iNFFT; ++i)
                    {
                        g_aiSumPowX[i] = *((int *) &g_pfSumPowX[i]);
                        g_aiSumPowY[i] = *((int *) &g_pfSumPowY[i]);
                        g_aiSumStokesRe[i] = *((int *) &g_pfSumStokesRe[i]);
                        g_aiSumStokesIm[i] = *((int *) &g_pfSumStokesIm[i]);
                    }
                    (void) write(iFileSpec, g_aiSumPowX, g_iNFFT * sizeof(int));
                    (void) write(iFileSpec, g_aiSumPowY, g_iNFFT * sizeof(int));
                    (void) write(iFileSpec, g_aiSumStokesRe, g_iNFFT * sizeof(int));
                    (void) write(iFileSpec, g_aiSumStokesIm, g_iNFFT * sizeof(int));
#endif

#if PLOT
                    /* NOTE: Plot() will modify data! */
                    Plot();
                    (void) usleep(500000);
#endif

                    /* reset time */
                    iTime = 0;
                    /* zero accumulators */
                    CUDASafeCallWithCleanUp(cudaMemset(g_pf4SumStokes_d,
                                                 '\0',
                                                 (g_iNumSubBands
                                                  * g_iNFFT
                                                  * sizeof(float4))));
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

        /* if time to read from input buffer */
        iProcData += (g_iNumSubBands * g_iNFFT * sizeof(char4));
        if ((g_iSizeRead
             - ((g_iNTaps - 1) * g_iNumSubBands * g_iNFFT * sizeof(char4)))
            == iProcData)
        {
            if (!(g_iIsDataReadDone))
            {
                /* read data from input buffer */
                iRet = ReadData();
                if (iRet != EXIT_SUCCESS)
                {
                    (void) fprintf(stderr, "ERROR: Data reading failed!\n");
                    break;
                }
                iProcData = 0;
            }
            else    /* no more data to be read */
            {
                g_iIsProcDone = TRUE;
            }
        }
#if BENCHMARKING
#endif
    }
#if (!BENCHMARKING)
    (void) gettimeofday(&stStop, NULL);
    (void) printf("Time taken (barring Init()): %gs\n",
                  ((stStop.tv_sec + (stStop.tv_usec * USEC2SEC))
                   - (stStart.tv_sec + (stStart.tv_usec * USEC2SEC))));
#endif

#if OUTFILE
    (void) close(iFileSpec);
#endif

    CleanUp();

#if BENCHMARKING
    PrintBenchmarks(fTotPFB,
                    iCountPFB,
                    fTotCpInFFT,
                    iCountCpInFFT,
                    fTotFFT,
                    iCountFFT,
                    fTotAccum,
                    iCountAccum,
                    fTotCpOut,
                    iCountCpOut);
    CUDASafeCallWithCleanUp(cudaEventDestroy(g_cuStart));
    CUDASafeCallWithCleanUp(cudaEventDestroy(g_cuStop));
    (void) printf("* Events destroyed.\n");
    (void) printf("* Benchmarking run completed.\n");
#endif

    return EXIT_SUCCESS;
}

/* function that creates the FFT plan, allocates memory, initialises counters,
   etc. */
int Init()
{
    int iDevCount = 0;
    cudaDeviceProp stDevProp = {0};
    int iRet = EXIT_SUCCESS;
    cufftResult iCUFFTRet = CUFFT_SUCCESS;
    int iFFTPlanN = g_iNFFT;
    int iFFTPlanINEmbed = g_iNFFT;
    int iFFTPlanONEmbed = g_iNFFT;

    iRet = RegisterSignalHandlers();
    if (iRet != EXIT_SUCCESS)
    {
        (void) fprintf(stderr, "ERROR: Signal-handler registration failed!\n");
        return EXIT_FAILURE;
    }

    CUDASafeCallWithCleanUp(cudaGetDeviceCount(&iDevCount));
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
    CUDASafeCallWithCleanUp(cudaSetDevice(0));

#if BENCHMARKING
    CUDASafeCallWithCleanUp(cudaEventCreate(&g_cuStart));
    CUDASafeCallWithCleanUp(cudaEventCreate(&g_cuStop));
    (void) printf("* Events created.\n");
#endif

    CUDASafeCallWithCleanUp(cudaGetDeviceProperties(&stDevProp, 0));
    g_iMaxThreadsPerBlock = stDevProp.maxThreadsPerBlock;

    if (g_iIsPFBOn)
    {
        /* set number of taps to NUM_TAPS if PFB is on, else number of
           taps = 1 */
        g_iNTaps = NUM_TAPS;
    }

    /* allocate memory for data array - 32MB is the block size for the VEGAS
       input buffer */
    CUDASafeCallWithCleanUp(cudaMalloc((void **) &g_pc4Data_d, g_iSizeRead));
    g_pc4DataRead_d = g_pc4Data_d;

    /* load data into memory */
    iRet = LoadDataToMem();
    if (iRet != EXIT_SUCCESS)
    {
        (void) fprintf(stderr, "ERROR! Loading to memory failed!\n");
        return EXIT_FAILURE;
    }

    /* calculate kernel parameters */
    if (g_iNFFT < g_iMaxThreadsPerBlock)
    {
        g_dimBPFB.x = g_iNFFT;
        g_dimBCopy.x = g_iNFFT;
        g_dimBAccum.x = g_iNFFT;
    }
    else
    {
        g_dimBPFB.x = g_iMaxThreadsPerBlock;
        g_dimBCopy.x = g_iMaxThreadsPerBlock;
        g_dimBAccum.x = g_iMaxThreadsPerBlock;
    }
    g_dimGPFB.x = (int) ceilf(((float) (g_iNumSubBands * g_iNFFT)) / g_iMaxThreadsPerBlock);
    g_dimGCopy.x = (int) ceilf(((float) (g_iNumSubBands * g_iNFFT)) / g_iMaxThreadsPerBlock);
    g_dimGAccum.x = (int) ceilf(((float) (g_iNumSubBands * g_iNFFT)) / g_iMaxThreadsPerBlock);

    iRet = ReadData();
    if (iRet != EXIT_SUCCESS)
    {
        (void) fprintf(stderr, "ERROR: Reading data failed!\n");
        return EXIT_FAILURE;
    }

    CUDASafeCallWithCleanUp(cudaMalloc((void **) &g_pf4FFTIn_d,
                                 g_iNumSubBands * g_iNFFT * sizeof(float4)));
    CUDASafeCallWithCleanUp(cudaMalloc((void **) &g_pf4FFTOut_d,
                                 g_iNumSubBands * g_iNFFT * sizeof(float4)));

    g_pf4SumStokes = (float4 *) malloc(g_iNumSubBands
                                       * g_iNFFT
                                       * sizeof(float4));
    if (NULL == g_pf4SumStokes)
    {
        (void) fprintf(stderr,
                       "ERROR: Memory allocation failed! %s.\n",
                       strerror(errno));
        return EXIT_FAILURE;
    }
    CUDASafeCallWithCleanUp(cudaMalloc((void **) &g_pf4SumStokes_d,
                                 g_iNumSubBands * g_iNFFT * sizeof(float4)));
    CUDASafeCallWithCleanUp(cudaMemset(g_pf4SumStokes_d,
                                 '\0',
                                 g_iNumSubBands * g_iNFFT * sizeof(float4)));

    /* create plan */
    iCUFFTRet = cufftPlanMany(&g_stPlan,
                              FFTPLAN_RANK,
                              &iFFTPlanN,
                              &iFFTPlanINEmbed,
                              FFTPLAN_ISTRIDE,
                              FFTPLAN_IDIST,
                              &iFFTPlanONEmbed,
                              FFTPLAN_OSTRIDE,
                              FFTPLAN_ODIST,
                              CUFFT_C2C,
                              FFTPLAN_BATCH);
    if (iCUFFTRet != CUFFT_SUCCESS)
    {
        (void) fprintf(stderr, "ERROR: Plan creation failed!\n");
        return EXIT_FAILURE;
    }

#if PLOT
    /* just for plotting */
    iRet = InitPlot();
    if (iRet != EXIT_SUCCESS)
    {
        (void) fprintf(stderr,
                       "ERROR: Plotting initialisation failed!\n");
        return EXIT_FAILURE;
    }
#endif

    return EXIT_SUCCESS;
}

/* function that reads data from the data file and loads it into memory during
   initialisation */
int LoadDataToMem()
{
    struct stat stFileStats = {0};
    int iRet = EXIT_SUCCESS;
    int iFileData = 0;

    iRet = stat(g_acFileData, &stFileStats);
    if (iRet != EXIT_SUCCESS)
    {
        (void) fprintf(stderr,
                       "ERROR: Failed to stat %s: %s!\n",
                       g_acFileData,
                       strerror(errno));
        (void) close(iFileData);
        return EXIT_FAILURE;
    }

    g_iSizeFile = stFileStats.st_size;
    g_pc4InBuf = (char4*) malloc(g_iSizeFile);
    if (NULL == g_pc4InBuf)
    {
        (void) fprintf(stderr,
                       "ERROR: Memory allocation failed! %s.\n",
                       strerror(errno));
        (void) close(iFileData);
        return EXIT_FAILURE;
    }

    iFileData = open(g_acFileData, O_RDONLY);
    if (EXIT_FAILURE == iFileData)
    {
        (void) fprintf(stderr,
                       "ERROR! Opening data file %s failed! %s.\n",
                       g_acFileData,
                       strerror(errno));
        return EXIT_FAILURE;
    }

    iRet = read(iFileData, g_pc4InBuf, g_iSizeFile);
    if (EXIT_FAILURE == iRet)
    {
        (void) fprintf(stderr,
                       "ERROR: Data reading failed! %s.\n",
                       strerror(errno));
        (void) close(iFileData);
        return EXIT_FAILURE;
    }

    (void) close(iFileData);
    
    /* set the read pointer to the beginning of the data array */
    g_pc4InBufRead = g_pc4InBuf;

    return EXIT_SUCCESS;
}

/* function that reads data from input buffer */
int ReadData()
{
    /* write new data to the write buffer */
#if BENCHMARKING
    CUDASafeCallWithCleanUp(cudaEventRecord(g_cuStart, 0));
    CUDASafeCallWithCleanUp(cudaEventSynchronize(g_cuStart));
#endif
    CUDASafeCallWithCleanUp(cudaMemcpy(g_pc4Data_d,
                                 g_pc4InBufRead,
                                 g_iSizeRead,
                                 cudaMemcpyHostToDevice));
#if BENCHMARKING
    CUDASafeCallWithCleanUp(cudaEventRecord(g_cuStop, 0));
    CUDASafeCallWithCleanUp(cudaEventSynchronize(g_cuStop));
    CUDASafeCallWithCleanUp(cudaEventElapsedTime(&g_fTimeCpIn, g_cuStart, g_cuStop));
    g_fTotCpIn += g_fTimeCpIn;
    ++g_iCountCpIn;
#endif
    /* update the read pointer to where data needs to be read in from, in the
       next read */
    g_pc4InBufRead += ((g_iSizeRead
                        - ((g_iNTaps - 1)
                           * g_iNumSubBands
                           * g_iNFFT
                           * sizeof(char4)))
                       / sizeof(char4));
    /* whenever there is a read, reset the read pointer to the beginning */
    g_pc4DataRead_d = g_pc4Data_d;
    ++g_iReadCount;
    /* TODO: won't read last block - check it */
    if ((((char *) g_pc4InBuf) + g_iSizeFile) - ((char *) g_pc4InBufRead)
        <= g_iSizeRead)
    {
        (void) printf("Data read done! Read count = %d\n", g_iReadCount);
        g_iIsDataReadDone = TRUE;
    }

    return EXIT_SUCCESS;
}

__global__ void DoPFB(char4* pc4Data,
                      float4* pf4FFTIn)
{
    int i = (blockIdx.x * blockDim.x) + threadIdx.x;
    int iNFFT = (gridDim.x * blockDim.x);
    int j = 0;
    int iAbsIdx = 0;
    float fArg = 0.0;
    float4 f4PFBOut = make_float4(0.0, 0.0, 0.0, 0.0);
    float fCoeff = 0.0;
    char4 c4Data = make_char4(0, 0, 0, 0);

    for (j = 0; j < NUM_TAPS; ++j)
    {
        /* calculate the absolute index */
        iAbsIdx = (j * iNFFT) + i;
        /* get the address of the block */
        c4Data = pc4Data[iAbsIdx];
        /* evaluate filter coefficient at this point */
        fArg = (float) M_PI * ((((float) iAbsIdx) / iNFFT) - (NUM_TAPS / 2));
        fCoeff = (((float) 0.0 == fArg) ? 1.0 : (__sinf(fArg) / fArg));
        
        f4PFBOut.x += (float) c4Data.x * fCoeff;
        f4PFBOut.y += (float) c4Data.y * fCoeff;
        f4PFBOut.z += (float) c4Data.z * fCoeff;
        f4PFBOut.w += (float) c4Data.w * fCoeff;
    }

    pf4FFTIn[i] = f4PFBOut;

    return;
}

__global__ void CopyDataForFFT(char4 *pc4Data,
                               float4 *pf4FFTIn)
{
    int i = (blockIdx.x * blockDim.x) + threadIdx.x;

    pf4FFTIn[i].x = (float) pc4Data[i].x;
    pf4FFTIn[i].y = (float) pc4Data[i].y;
    pf4FFTIn[i].z = (float) pc4Data[i].z;
    pf4FFTIn[i].w = (float) pc4Data[i].w;

    return;
}

/* function that performs the FFT */
int DoFFT()
{
    cufftResult iCUFFTRet = CUFFT_SUCCESS;

    /* execute plan */
    iCUFFTRet = cufftExecC2C(g_stPlan,
                             (cufftComplex*) g_pf4FFTIn_d,
                             (cufftComplex*) g_pf4FFTOut_d,
                             CUFFT_FORWARD);
    if (iCUFFTRet != CUFFT_SUCCESS)
    {
        (void) fprintf(stderr, "ERROR! FFT for polarisation X failed!\n");
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}

__global__ void Accumulate(float4 *pf4FFTOut,
                           float4 *pf4SumStokes)
{
    int i = (blockIdx.x * blockDim.x) + threadIdx.x;

    /* Re(X)^2 + Im(X)^2 */
    pf4SumStokes[i].x += (pf4FFTOut[i].x * pf4FFTOut[i].x)
                         + (pf4FFTOut[i].y * pf4FFTOut[i].y);
    /* Re(Y)^2 + Im(Y)^2 */
    pf4SumStokes[i].y += (pf4FFTOut[i].z * pf4FFTOut[i].z)
                         + (pf4FFTOut[i].w * pf4FFTOut[i].w);
    /* Re(XY*) */
    pf4SumStokes[i].z += (pf4FFTOut[i].x * pf4FFTOut[i].z)
                         + (pf4FFTOut[i].y * pf4FFTOut[i].w);
    /* Im(XY*) */
    pf4SumStokes[i].w += (pf4FFTOut[i].y * pf4FFTOut[i].z)
                         - (pf4FFTOut[i].x * pf4FFTOut[i].w);

    return;
}

int IsRunning()
{
    return (!g_iIsProcDone);
}

int IsBlankingSet()
{
    /* check for status and return TRUE or FALSE */
    return FALSE;
}

/* function that frees resources */
void CleanUp()
{
    /* free memory */
    if (g_pc4InBuf != NULL)
    {
        free(g_pc4InBuf);
        g_pc4InBuf = NULL;
    }
    if (g_pc4Data_d != NULL)
    {
        (void) cudaFree(g_pc4Data_d);
        g_pc4Data_d = NULL;
    }
    if (g_pf4FFTIn_d != NULL)
    {
        (void) cudaFree(g_pf4FFTIn_d);
        g_pf4FFTIn_d = NULL;
    }
    if (g_pf4FFTOut_d != NULL)
    {
        (void) cudaFree(g_pf4FFTOut_d);
        g_pf4FFTOut_d = NULL;
    }
    if (g_pf4SumStokes != NULL)
    {
        free(g_pf4SumStokes);
        g_pf4SumStokes = NULL;
    }
    if (g_pf4SumStokes_d != NULL)
    {
        (void) cudaFree(g_pf4SumStokes_d);
        g_pf4SumStokes_d = NULL;
    }

    /* destroy plan */
    (void) cufftDestroy(g_stPlan);

#if PLOT
    if (g_pfSumPowX != NULL)
    {
        free(g_pfSumPowX);
        g_pfSumPowX = NULL;
    }
    if (g_pfSumPowY != NULL)
    {
        free(g_pfSumPowY);
        g_pfSumPowY = NULL;
    }
    if (g_pfSumStokesRe != NULL)
    {
        free(g_pfSumStokesRe);
        g_pfSumStokesRe = NULL;
    }
    if (g_pfSumStokesIm != NULL)
    {
        free(g_pfSumStokesIm);
        g_pfSumStokesIm = NULL;
    }
    if (g_pfFreq != NULL)
    {
        free(g_pfFreq);
        g_pfFreq = NULL;
    }
    cpgclos();
#endif

    return;
}

#if BENCHMARKING
/* function to print benchmarking statistics */
void PrintBenchmarks(float fTotPFB,
                     int iCountPFB,
                     float fTotCpInFFT,
                     int iCountCpInFFT,
                     float fTotFFT,
                     int iCountFFT,
                     float fTotAccum,
                     int iCountAccum,
                     float fTotCpOut,
                     int iCountCpOut)
{
    float fTotal = 0.0;
    
    fTotal = g_fTotCpIn
             + fTotPFB
             + fTotCpInFFT
             + fTotFFT
             + fTotAccum
             + fTotCpOut;
    (void) printf("    Total elapsed time for\n");
    (void) printf("        %6d calls to cudaMemcpy(Host2Device)          : "
                  "%5.3fms, %2d%%; Average = %5.3fms\n",
                  g_iCountCpIn,
                  g_fTotCpIn,
                  (int) ((g_fTotCpIn / fTotal) * 100),
                  g_fTotCpIn / g_iCountCpIn);
    if (g_iIsPFBOn)
    {
        (void) printf("        %6d calls to DoPFB()                          : "
                      "%5.3fms, %2d%%; Average = %5.3fms\n",
                      iCountPFB,
                      fTotPFB,
                      (int) ((fTotPFB / fTotal) * 100),
                      fTotPFB / iCountPFB);
    }
    else
    {
        (void) printf("        %6d calls to CopyDataForFFT()                 : "
                      "%5.3fms, %2d%%; Average = %5.3fms\n",
                      iCountCpInFFT,
                      fTotCpInFFT,
                      (int) ((fTotCpInFFT / fTotal) * 100),
                      fTotCpInFFT / iCountCpInFFT);
    }
    (void) printf("        %6d calls to DoFFT()                          : "
                  "%5.3fms, %2d%%; Average = %5.3fms\n",
                  iCountFFT,
                  fTotFFT,
                  (int) ((fTotFFT / fTotal) * 100),
                  fTotFFT / iCountFFT);
    (void) printf("        %6d calls to Accumulate()/accumulation loop   : "
                  "%5.3fms, %2d%%; Average = %5.3fms\n",
                  iCountAccum,
                  fTotAccum,
                  (int) ((fTotAccum / fTotal) * 100),
                  fTotAccum / iCountAccum);
    (void) printf("        %6d calls to cudaMemcpy(Device2Host)          : "
                  "%5.3fms, %2d%%; Average = %5.3fms\n",
                  iCountCpOut,
                  fTotCpOut,
                  (int) ((fTotCpOut / fTotal) * 100),
                  fTotCpOut / iCountCpOut);

    return;
}
#endif

#if PLOT
int InitPlot()
{
    int iRet = EXIT_SUCCESS;
    int i = 0;

    iRet = cpgopen(PG_DEV);
    if (iRet <= 0)
    {
        (void) fprintf(stderr,
                       "ERROR: Opening graphics device %s failed!\n",
                       PG_DEV);
        return EXIT_FAILURE;
    }

    cpgsch(3);
    cpgsubp(g_iNumSubBands, 4);

    g_pfSumPowX = (float*) malloc(g_iNFFT * sizeof(float));
    if (NULL == g_pfSumPowX)
    {
        (void) fprintf(stderr,
                       "ERROR: Memory allocation failed! %s.\n",
                       strerror(errno));
        return EXIT_FAILURE;
    }
    g_pfSumPowY = (float*) malloc(g_iNFFT * sizeof(float));
    if (NULL == g_pfSumPowY)
    {
        (void) fprintf(stderr,
                       "ERROR: Memory allocation failed! %s.\n",
                       strerror(errno));
        return EXIT_FAILURE;
    }
    g_pfSumStokesRe = (float*) malloc(g_iNFFT * sizeof(float));
    if (NULL == g_pfSumStokesRe)
    {
        (void) fprintf(stderr,
                       "ERROR: Memory allocation failed! %s.\n",
                       strerror(errno));
        return EXIT_FAILURE;
    }
    g_pfSumStokesIm = (float*) malloc(g_iNFFT * sizeof(float));
    if (NULL == g_pfSumStokesIm)
    {
        (void) fprintf(stderr,
                       "ERROR: Memory allocation failed! %s.\n",
                       strerror(errno));
        return EXIT_FAILURE;
    }
    g_pfFreq = (float*) malloc(g_iNFFT * sizeof(float));
    if (NULL == g_pfFreq)
    {
        (void) fprintf(stderr,
                       "ERROR: Memory allocation failed! %s.\n",
                       strerror(errno));
        return EXIT_FAILURE;
    }

    /* load the frequency axis */
    for (i = 0; i < g_iNFFT; ++i)
    {
        g_pfFreq[i] = ((float) i * g_fFSamp) / g_iNFFT;
    }

    return EXIT_SUCCESS;
}

void Plot()
{
    float fMinFreq = g_pfFreq[0];
    float fMaxFreq = g_pfFreq[g_iNFFT-1];
    float fMinY = FLT_MAX;
    float fMaxY = -(FLT_MAX);
    int i = 0;
    int j = 0;
    int k = 0;

    for (k = 0; k < g_iNumSubBands; ++k)
    {
        for (i = k, j = 0; i < (g_iNumSubBands * g_iNFFT); i += g_iNumSubBands, ++j)
        {
            if (0.0 == g_pf4SumStokes[i].x)
            {
                g_pfSumPowX[j] = 0.0;
            }
            else
            {
                g_pfSumPowX[j] = 10 * log10f(g_pf4SumStokes[i].x);
            }
            if (0.0 == g_pf4SumStokes[i].y)
            {
                g_pfSumPowY[j] = 0.0;
            }
            else
            {
                g_pfSumPowY[j] = 10 * log10f(g_pf4SumStokes[i].y);
            }
            g_pfSumStokesRe[j] = g_pf4SumStokes[i].z;
            g_pfSumStokesIm[j] = g_pf4SumStokes[i].w;
        }

        /* plot accumulated X-pol. power */
        fMinY = FLT_MAX;
        fMaxY = -(FLT_MAX);
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
        /* to avoid min == max */
        fMaxY += 1.0;
        fMinY -= 1.0;
        #if 1
        for (i = 0; i < g_iNFFT; ++i)
        {
            g_pfSumPowX[i] -= fMaxY;
        }
        fMinY -= fMaxY;
        fMaxY = 0;
        #endif
        cpgpanl(k + 1, 1);
        cpgeras();
        cpgsvp(PG_VP_ML, PG_VP_MR, PG_VP_MB, PG_VP_MT);
        cpgswin(fMinFreq, fMaxFreq, fMinY, fMaxY);
        //cpglab("Bin Number", "", "SumPowX");
        cpgbox("BCNST", 0.0, 0, "BCNST", 0.0, 0);
        cpgsci(PG_CI_PLOT);
        cpgline(g_iNFFT, g_pfFreq, g_pfSumPowX);
        cpgsci(PG_CI_DEF);

        /* plot accumulated Y-pol. power */
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
        /* to avoid min == max */
        fMaxY += 1.0;
        fMinY -= 1.0;
        #if 1
        for (i = 0; i < g_iNFFT; ++i)
        {
            g_pfSumPowY[i] -= fMaxY;
        }
        fMinY -= fMaxY;
        fMaxY = 0;
        #endif
        cpgpanl(k + 1, 2);
        cpgeras();
        cpgsvp(PG_VP_ML, PG_VP_MR, PG_VP_MB, PG_VP_MT);
        cpgswin(fMinFreq, fMaxFreq, fMinY, fMaxY);
        //cpglab("Bin Number", "", "SumPowY");
        cpgbox("BCNST", 0.0, 0, "BCNST", 0.0, 0);
        cpgsci(PG_CI_PLOT);
        cpgline(g_iNFFT, g_pfFreq, g_pfSumPowY);
        cpgsci(PG_CI_DEF);

        /* plot accumulated real(XY*) */
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
        /* to avoid min == max */
        fMaxY += 1.0;
        fMinY -= 1.0;
        cpgpanl(k + 1, 3);
        cpgeras();
        cpgsvp(PG_VP_ML, PG_VP_MR, PG_VP_MB, PG_VP_MT);
        cpgswin(fMinFreq, fMaxFreq, fMinY, fMaxY);
        //cpglab("Bin Number", "", "SumStokesRe");
        cpgbox("BCNST", 0.0, 0, "BCNST", 0.0, 0);
        cpgsci(PG_CI_PLOT);
        cpgline(g_iNFFT, g_pfFreq, g_pfSumStokesRe);
        cpgsci(PG_CI_DEF);

        /* plot accumulated imag(XY*) */
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
        /* to avoid min == max */
        fMaxY += 1.0;
        fMinY -= 1.0;
        cpgpanl(k + 1, 4);
        cpgeras();
        cpgsvp(PG_VP_ML, PG_VP_MR, PG_VP_MB, PG_VP_MT);
        cpgswin(fMinFreq, fMaxFreq, fMinY, fMaxY);
        //cpglab("Bin Number", "", "SumStokesIm");
        cpgbox("BCNST", 0.0, 0, "BCNST", 0.0, 0);
        cpgsci(PG_CI_PLOT);
        cpgline(g_iNFFT, g_pfFreq, g_pfSumStokesIm);
        cpgsci(PG_CI_DEF);
    }
   // CleanUp();
   // exit(0);

    return;
}
#endif

/*
 * Registers handlers for SIGTERM and CTRL+C
 */
int RegisterSignalHandlers()
{
    struct sigaction stSigHandler = {{0}};
    int iRet = EXIT_SUCCESS;

    /* register the CTRL+C-handling function */
    stSigHandler.sa_handler = HandleStopSignals;
    iRet = sigaction(SIGINT, &stSigHandler, NULL);
    if (iRet != EXIT_SUCCESS)
    {
        (void) fprintf(stderr,
                       "ERROR: Handler registration failed for signal %d!\n",
                       SIGINT);
        return EXIT_FAILURE;
    }

    /* register the SIGTERM-handling function */
    stSigHandler.sa_handler = HandleStopSignals;
    iRet = sigaction(SIGTERM, &stSigHandler, NULL);
    if (iRet != EXIT_SUCCESS)
    {
        (void) fprintf(stderr,
                       "ERROR: Handler registration failed for signal %d!\n",
                       SIGTERM);
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}

/*
 * Catches SIGTERM and CTRL+C and cleans up before exiting
 */
void HandleStopSignals(int iSigNo)
{
    /* clean up */
    CleanUp();

    /* exit */
    exit(EXIT_SUCCESS);

    /* never reached */
    return;
}

void __CUDASafeCallWithCleanUp(cudaError_t iRet,
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
        exit(EXIT_FAILURE);
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

