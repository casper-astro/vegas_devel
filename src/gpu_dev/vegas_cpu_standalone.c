/** 
 * @file vegas_cpu_standalone.c
 * VEGAS Low-Bandwidth Modes - Stand-Alone CPU Implementation
 *
 * @author Jayanth Chennamangalam
 * @date 2011.06.02
 * Modified to include Mathew Bailes' optimisations on 2011.06.21
 */

#include "vegas_cpu_standalone.h"

int g_iIsDone = FALSE;
signed char *g_pcInBuf = NULL;
int g_iSizeFile = 0;
int g_iReadCount = 0;
int g_iNumReads = 0;
PFB_DATA g_astPFBData[NUM_TAPS] = {{0}};
int g_iPFBReadIdx = 0;
int g_iPFBWriteIdx = 0;
int g_iNFFT = DEF_LEN_SPEC;
fftwf_complex *g_pfcFFTInX = NULL;
fftwf_complex *g_pfcFFTOutX = NULL;
fftwf_plan g_stPlanX = {0};
fftwf_complex *g_pfcFFTInY = NULL;
fftwf_complex *g_pfcFFTOutY = NULL;
fftwf_plan g_stPlanY = {0};
float *g_pfSumPowX = NULL;
float *g_pfSumPowY = NULL;
float *g_pfSumStokesRe = NULL;
float *g_pfSumStokesIm = NULL;
int g_iIsPFBOn = DEF_PFB_ON;
int g_iNTaps = 1;                       /* 1 if no PFB, NUM_TAPS if PFB */
char g_acFileData[256] = {0};
/* BUG: crash if file size is less than 32MB */
int g_iSizeRead = DEF_SIZE_READ;
int g_iFileCoeff = 0;
char g_acFileCoeff[256] = {0};
signed char *g_pcPFBCoeff = NULL;
signed char *g_pcOptPFBCoeff = NULL;
float g_afConvLookup[256] = {0};
int g_iFileData = 0;

#if PLOT
float* g_pfFreq = NULL;
float g_fFSamp = 1.0;                   /* 1 [frequency] */
#endif

int main(int argc, char *argv[])
{
    int iRet = EXIT_SUCCESS;
    int i = 0;
    int iSpecCount = 0;
    int iNumAcc = DEF_ACC;
    struct timeval stStart = {0};
    struct timeval stStop = {0};
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
                iNumAcc = (int) atoi(optarg);
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
    if (iFileSpec < EXIT_SUCCESS)
    {
        (void) fprintf(stderr, "ERROR! Opening spectrum file failed!\n");
        CleanUp();
        return EXIT_FAILURE;
    }
#endif

    (void) gettimeofday(&stStart, NULL);
    while (!g_iIsDone)
    {
        if (g_iIsPFBOn)
        {
            /* do pfb */
            (void) DoPFB();
        }
        else
        {
            /* copy data for FFT */
            (void) CopyDataForFFT();
        }

        /* do fft */
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

        /* accumulate power x, power y, stokes, if the blanking bit is
           not set */
        for (i = 0; i < g_iNFFT; ++i)
        {
            /* Re(X)^2 + Im(X)^2 */
            g_pfSumPowX[i] += (g_pfcFFTOutX[i][0] * g_pfcFFTOutX[i][0])
                              + (g_pfcFFTOutX[i][1] * g_pfcFFTOutX[i][1]);
            /* Re(Y)^2 + Im(Y)^2 */
            g_pfSumPowY[i] += (g_pfcFFTOutY[i][0] * g_pfcFFTOutY[i][0])
                              + (g_pfcFFTOutY[i][1] * g_pfcFFTOutY[i][1]);
            /* Re(XY*) */
            g_pfSumStokesRe[i] += (g_pfcFFTOutX[i][0] * g_pfcFFTOutY[i][0])
                                  + (g_pfcFFTOutX[i][1] * g_pfcFFTOutY[i][1]);
            /* Im(XY*) */
            g_pfSumStokesIm[i] += (g_pfcFFTOutX[i][1] * g_pfcFFTOutY[i][0])
                                  - (g_pfcFFTOutX[i][0] * g_pfcFFTOutY[i][1]);
        }
        ++iSpecCount;
        if (iSpecCount == iNumAcc)
        {
#if OUTFILE
            (void) write(iFileSpec, g_pfSumPowX, g_iNFFT * sizeof(float));
            (void) write(iFileSpec, g_pfSumPowY, g_iNFFT * sizeof(float));
            (void) write(iFileSpec, g_pfSumStokesRe, g_iNFFT * sizeof(float));
            (void) write(iFileSpec, g_pfSumStokesIm, g_iNFFT * sizeof(float));
#endif

#if PLOT
            /* NOTE: Plot() will modify data! */
            Plot();
            (void) usleep(500000);
#endif

            /* reset time */
            iSpecCount = 0;
            /* zero accumulators */
            (void) memset(g_pfSumPowX, '\0', g_iNFFT * sizeof(float));
            (void) memset(g_pfSumPowY, '\0', g_iNFFT * sizeof(float));
            (void) memset(g_pfSumStokesRe, '\0', g_iNFFT * sizeof(float));
            (void) memset(g_pfSumStokesIm, '\0', g_iNFFT * sizeof(float));
        }

        /* read data from input buffer */
        iRet = ReadData();
        if (iRet != EXIT_SUCCESS)
        {
            (void) fprintf(stderr, "ERROR: Data reading failed!\n");
            break;
        }
    }
    (void) gettimeofday(&stStop, NULL);
    (void) printf("Time taken (barring Init()): %gs\n",
                  ((stStop.tv_sec + (stStop.tv_usec * USEC2SEC))
                   - (stStart.tv_sec + (stStart.tv_usec * USEC2SEC))));

#if OUTFILE
    (void) close(iFileSpec);
#endif

    CleanUp();

    return EXIT_SUCCESS;
}

/* function that creates the FFT plan, allocates memory, initialises counters,
   etc. */
int Init()
{
    int i = 0;
    int j = 0;
    int k = 0;
    int iRet = EXIT_SUCCESS;

    iRet = RegisterSignalHandlers();
    if (iRet != EXIT_SUCCESS)
    {
        (void) fprintf(stderr, "ERROR: Signal-handler registration failed!\n");
        return EXIT_FAILURE;
    }

    if (g_iIsPFBOn)
    {
        /* set number of taps to NUM_TAPS if PFB is on, else number of
           taps = 1 */
        g_iNTaps = NUM_TAPS;

        g_pcPFBCoeff = (signed char *) malloc(g_iNTaps
                                              * g_iNFFT
                                              * sizeof(signed char));
        if (NULL == g_pcPFBCoeff)
        {
            (void) fprintf(stderr,
                           "ERROR: Memory allocation failed! %s.\n",
                           strerror(errno));
            return EXIT_FAILURE;
        }
		g_pcOptPFBCoeff = (signed char *) malloc(2
                                                 * g_iNTaps
                                                 * g_iNFFT
                                                 * sizeof(signed char)); 
        if (NULL == g_pcOptPFBCoeff)
        {
            (void) fprintf(stderr,
                           "ERROR: Memory allocation failed! %s.\n",
                           strerror(errno));
            return EXIT_FAILURE;
        }

        /* read filter coefficients */
        /* build file name */
        (void) sprintf(g_acFileCoeff,
                       "%s_%s_%d_%d_%d%s",
                       FILE_COEFF_PREFIX,
                       FILE_COEFF_DATATYPE,
                       g_iNTaps,
                       g_iNFFT,
                       DEF_NUM_SUBBANDS,
                       FILE_COEFF_SUFFIX);
        g_iFileCoeff = open(g_acFileCoeff, O_RDONLY);
        if (g_iFileCoeff < EXIT_SUCCESS)
        {
            (void) fprintf(stderr,
                           "ERROR: Opening filter coefficients file %s "
                           "failed! %s.\n",
                           g_acFileCoeff,
                           strerror(errno));
            return EXIT_FAILURE;
        }

        iRet = read(g_iFileCoeff,
                    g_pcPFBCoeff,
                    g_iNTaps * g_iNFFT * sizeof(signed char));
        if (iRet != (g_iNTaps * g_iNFFT * sizeof(signed char)))
        {
            (void) fprintf(stderr,
                           "ERROR: Reading filter coefficients failed! %s.\n",
                           strerror(errno));
            return EXIT_FAILURE;
        }
        (void) close(g_iFileCoeff);

    	/* duplicate the coefficients for PFB optimisation */
        for (i = 0; i < (g_iNTaps * g_iNFFT); ++i)
        {
            g_pcOptPFBCoeff[2*i] = g_pcPFBCoeff[i];
            g_pcOptPFBCoeff[(2*i)+1] = g_pcPFBCoeff[i];
    	}
    }

    /* allocate memory for data array contents */
    for (i = 0; i < g_iNTaps; ++i)
    {
        g_astPFBData[i].pfcDataX = (fftwf_complex *) fftwf_malloc(g_iNFFT
                                                      * sizeof(fftwf_complex));
        if (NULL == g_astPFBData[i].pfcDataX)
        {
            (void) fprintf(stderr,
                           "ERROR: Memory allocation failed! %s.\n",
                           strerror(errno));
            return EXIT_FAILURE;
        }
        g_astPFBData[i].pfcDataY = (fftwf_complex *) fftwf_malloc(g_iNFFT
                                                      * sizeof(fftwf_complex));
        if (NULL == g_astPFBData[i].pfcDataY)
        {
            (void) fprintf(stderr,
                           "ERROR: Memory allocation failed! %s.\n",
                           strerror(errno));
            return EXIT_FAILURE;
        }
        if (i != (g_iNTaps - 1))
        {
            g_astPFBData[i].iNextIdx = i + 1;
        }
        else
        {
            g_astPFBData[i].iNextIdx = 0;
        }
    }

    g_iFileData = open(g_acFileData, O_RDONLY);
    if (g_iFileData < EXIT_SUCCESS)
    {
        (void) fprintf(stderr,
                       "ERROR! Opening data file %s failed! %s.\n",
                       g_acFileData,
                       strerror(errno));
        return EXIT_FAILURE;
    }

    /* load data into memory */
    iRet = LoadDataToMem();
    if (iRet != EXIT_SUCCESS)
    {
        (void) fprintf(stderr, "ERROR! Loading to memory failed!\n");
        return EXIT_FAILURE;
    }

    /* read first g_iNTaps blocks of data */
    /* ASSUMPTION: there is at least g_iNTaps blocks of data */
    for (i = 0; i < g_iNTaps; ++i)
    {
        g_astPFBData[i].pcData = g_pcInBuf + (i * LEN_DATA);
        ++g_iReadCount;
        if (g_iReadCount == g_iNumReads)
        {
            (void) printf("Data read done!\n");
            g_iIsDone = TRUE;
        }

        /* unpack data */
        /* assuming real and imaginary parts are interleaved, and X and Y are
           interleaved, like so:
           reX, imX, reY, imY, ... */
        j = 0;
        for (k = 0; k < LEN_DATA; k += NUM_BYTES_PER_SAMP)
        {
            g_astPFBData[i].pfcDataX[j][0]
                                    = (float) g_astPFBData[i].pcData[k];
            g_astPFBData[i].pfcDataX[j][1]
                                    = (float) g_astPFBData[i].pcData[k+1];
            g_astPFBData[i].pfcDataY[j][0]
                                    = (float) g_astPFBData[i].pcData[k+2];
            g_astPFBData[i].pfcDataY[j][1]
                                    = (float) g_astPFBData[i].pcData[k+3];
            ++j;
        }
    }
    g_iPFBWriteIdx = 0;     /* next write into the first buffer */
    g_iPFBReadIdx = 0;      /* PFB to be performed from first buffer */

    g_pfcFFTInX = (fftwf_complex *) fftwf_malloc(g_iNFFT
                                                 * sizeof(fftwf_complex));
    if (NULL == g_pfcFFTInX)
    {
        (void) fprintf(stderr,
                       "ERROR: Memory allocation failed! %s.\n",
                       strerror(errno));
        return EXIT_FAILURE;
    }
    g_pfcFFTInY = (fftwf_complex *) fftwf_malloc(g_iNFFT
                                                 * sizeof(fftwf_complex));
    if (NULL == g_pfcFFTInY)
    {
        (void) fprintf(stderr,
                       "ERROR: Memory allocation failed! %s.\n",
                       strerror(errno));
        return EXIT_FAILURE;
    }
    g_pfcFFTOutX = (fftwf_complex *) fftwf_malloc(g_iNFFT
                                                  * sizeof(fftwf_complex));
    if (NULL == g_pfcFFTOutX)
    {
        (void) fprintf(stderr,
                       "ERROR: Memory allocation failed! %s.\n",
                       strerror(errno));
        return EXIT_FAILURE;
    }
    g_pfcFFTOutY = (fftwf_complex *) fftwf_malloc(g_iNFFT
                                                  * sizeof(fftwf_complex));
    if (NULL == g_pfcFFTOutY)
    {
        (void) fprintf(stderr,
                       "ERROR: Memory allocation failed! %s.\n",
                       strerror(errno));
        return EXIT_FAILURE;
    }

    g_pfSumPowX = (float *) calloc(g_iNFFT, sizeof(float));
    if (NULL == g_pfSumPowX)
    {
        (void) fprintf(stderr,
                       "ERROR: Memory allocation failed! %s.\n",
                       strerror(errno));
        return EXIT_FAILURE;
    }
    g_pfSumPowY = (float *) calloc(g_iNFFT, sizeof(float));
    if (NULL == g_pfSumPowY)
    {
        (void) fprintf(stderr,
                       "ERROR: Memory allocation failed! %s.\n",
                       strerror(errno));
        return EXIT_FAILURE;
    }
    g_pfSumStokesRe = (float *) calloc(g_iNFFT, sizeof(float));
    if (NULL == g_pfSumStokesRe)
    {
        (void) fprintf(stderr,
                       "ERROR: Memory allocation failed! %s.\n",
                       strerror(errno));
        return EXIT_FAILURE;
    }
    g_pfSumStokesIm = (float *) calloc(g_iNFFT, sizeof(float));
    if (NULL == g_pfSumStokesIm)
    {
        (void) fprintf(stderr,
                       "ERROR: Memory allocation failed! %s.\n",
                       strerror(errno));
        return EXIT_FAILURE;
    }

    /* create plans */
    g_stPlanX = fftwf_plan_dft_1d(g_iNFFT,
                                 g_pfcFFTInX,
                                 g_pfcFFTOutX,
                                 FFTW_FORWARD,
                                 FFTW_MEASURE);
    g_stPlanY = fftwf_plan_dft_1d(g_iNFFT,
                                 g_pfcFFTInY,
                                 g_pfcFFTOutY,
                                 FFTW_FORWARD,
                                 FFTW_MEASURE);

#if PLOT
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

    iRet = stat(g_acFileData, &stFileStats);
    if (iRet != EXIT_SUCCESS)
    {
        (void) fprintf(stderr,
                       "ERROR: Failed to stat %s: %s!\n",
                       g_acFileData,
                       strerror(errno));
        return EXIT_FAILURE;
    }

    g_pcInBuf = (signed char*) malloc(stFileStats.st_size
                                      * sizeof(signed char));
    if (NULL == g_pcInBuf)
    {
        (void) fprintf(stderr,
                       "ERROR: Memory allocation failed! %s.\n",
                       strerror(errno));
        return EXIT_FAILURE;
    }

    iRet = read(g_iFileData, g_pcInBuf, stFileStats.st_size);
    if (iRet < EXIT_SUCCESS)
    {
        (void) fprintf(stderr,
                       "ERROR: Data reading failed! %s.\n",
                       strerror(errno));
        return EXIT_FAILURE;
    }
    else if (iRet != stFileStats.st_size)
    {
        (void) printf("File read done!\n");
    }

    /* calculate the number of reads required */
    g_iNumReads = stFileStats.st_size / LEN_DATA;

    return EXIT_SUCCESS;
}

/* function that reads data from input buffer */
int ReadData()
{
    int i = 0;
    int j = 0;

    /* write new data to the write buffer */
    g_astPFBData[g_iPFBWriteIdx].pcData += (g_iNTaps * LEN_DATA);
    ++g_iReadCount;
    if (g_iReadCount == g_iNumReads)
    {
        (void) printf("Data read done!\n");
        g_iIsDone = TRUE;
    }

    /* unpack data */
    /* assuming real and imaginary parts are interleaved, and X and Y are
       interleaved, like so:
       Re(X), Im(X), Re(Y), Im(Y), ... */
    j = 0;
    for (i = 0; i < LEN_DATA; i += NUM_BYTES_PER_SAMP)
    {
        g_astPFBData[g_iPFBWriteIdx].pfcDataX[j][0]
                            = (float) g_astPFBData[g_iPFBWriteIdx].pcData[i];
        g_astPFBData[g_iPFBWriteIdx].pfcDataX[j][1]
                            = (float) g_astPFBData[g_iPFBWriteIdx].pcData[i+1];
        g_astPFBData[g_iPFBWriteIdx].pfcDataY[j][0]
                            = (float) g_astPFBData[g_iPFBWriteIdx].pcData[i+2];
        g_astPFBData[g_iPFBWriteIdx].pfcDataY[j][1]
                            = (float) g_astPFBData[g_iPFBWriteIdx].pcData[i+3];
        ++j;
    }

    g_iPFBWriteIdx = g_astPFBData[g_iPFBWriteIdx].iNextIdx;
    g_iPFBReadIdx = g_astPFBData[g_iPFBReadIdx].iNextIdx;

    return EXIT_SUCCESS;
}

/* function that performs the PFB */
int DoPFB()
{
    int i = 0;
    int j = 0;
    int k = 0;
    int iCoeffStartIdx = 0;
	float *pfIn = NULL;
	signed char *pcCoeff = NULL;
	float *pfOut = NULL;

    /* reset memory */
    (void) memset(g_pfcFFTInX, '\0', g_iNFFT * sizeof(fftwf_complex));
    (void) memset(g_pfcFFTInY, '\0', g_iNFFT * sizeof(fftwf_complex));

    i = g_iPFBReadIdx;
    for (j = 0; j < g_iNTaps; ++j)
    {
        iCoeffStartIdx = 2 * j * g_iNFFT;
		pfOut = &g_pfcFFTInX[0][0];
		pfIn = &g_astPFBData[i].pfcDataX[0][0];
		pcCoeff = &g_pcOptPFBCoeff[iCoeffStartIdx];
        for (k = 0; k < (2 * g_iNFFT); ++k)
        {
            pfOut[k] += pfIn[k] * pcCoeff[k];
        }
		pfOut = &g_pfcFFTInY[0][0];
		pfIn = &g_astPFBData[i].pfcDataY[0][0];
		pcCoeff = &g_pcOptPFBCoeff[iCoeffStartIdx];
        for (k = 0; k < (2 * g_iNFFT); ++k)
		{
	  		pfOut[k] += pfIn[k] * pcCoeff[k];
		}
        i = g_astPFBData[i].iNextIdx;
    }

    return EXIT_SUCCESS;
}

int CopyDataForFFT()
{
    (void) memcpy(g_pfcFFTInX,
                  g_astPFBData[g_iPFBReadIdx].pfcDataX,
                  g_iNFFT * sizeof(fftwf_complex));
    (void) memcpy(g_pfcFFTInY,
                  g_astPFBData[g_iPFBReadIdx].pfcDataY,
                  g_iNFFT * sizeof(fftwf_complex));

    return EXIT_SUCCESS;
}

/* function that performs the FFT */
int DoFFT()
{
    /* execute plan */
    fftwf_execute(g_stPlanX);
    fftwf_execute(g_stPlanY);

    return EXIT_SUCCESS;
}

/* function that frees resources */
void CleanUp()
{
    int i = 0;

    /* free resources */
    free(g_pcInBuf);

    for (i = 0; i < g_iNTaps; ++i)
    {
        fftwf_free(g_astPFBData[i].pfcDataX);
        fftwf_free(g_astPFBData[i].pfcDataY);
    }
    fftwf_free(g_pfcFFTInX);
    fftwf_free(g_pfcFFTInY);
    fftwf_free(g_pfcFFTOutX);
    fftwf_free(g_pfcFFTOutY);

    free(g_pcPFBCoeff);
	free(g_pcOptPFBCoeff);

    free(g_pfSumPowX);
    free(g_pfSumPowY);
    free(g_pfSumStokesRe);
    free(g_pfSumStokesIm);

    /* destroy plans */
    fftwf_destroy_plan(g_stPlanX);
    fftwf_destroy_plan(g_stPlanY);

    fftwf_cleanup();

    (void) close(g_iFileData);

#if PLOT
    /* for plotting */
    free(g_pfFreq);
    cpgclos();
#endif

    return;
}

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
    cpgsubp(DEF_NUM_SUBBANDS, 4);

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

    for (i = 0; i < g_iNFFT; ++i)
    {
        if (g_pfSumPowX[i] != 0.0)
        {
            g_pfSumPowX[i] = 10 * log10f(g_pfSumPowX[i]);
        }
        if (g_pfSumPowY[i] != 0.0)
        {
            g_pfSumPowY[i] = 10 * log10f(g_pfSumPowY[i]);
        }
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
    cpgpanl(1, 1);
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
    cpgpanl(1, 2);
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
    cpgpanl(1, 3);
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
    cpgpanl(1, 4);
    cpgeras();
    cpgsvp(PG_VP_ML, PG_VP_MR, PG_VP_MB, PG_VP_MT);
    cpgswin(fMinFreq, fMaxFreq, fMinY, fMaxY);
    //cpglab("Bin Number", "", "SumStokesIm");
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

