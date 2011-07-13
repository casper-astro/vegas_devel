/* 
 * vegasp3mcpu.byte.c
 * VEGAS Priority 3 Mode - Stand-Alone CPU Implementation
 *
 * Created by Jayanth Chennamangalam on 2011.06.02
 * Modified to include Mathew Bailes' optimisations on 2011.06.21
 */

#include "vegasp3mcpu.byte.h"

int g_iIsDone = FALSE;

BYTE *g_pbInBuf = NULL;
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

#if OUTFILE
int g_aiSumPowX[2048] = {0};
int g_aiSumPowY[2048] = {0};
int g_aiSumStokesRe[2048] = {0};
int g_aiSumStokesIm[2048] = {0};
#endif

int g_iIsPFBOn = DEF_PFB_ON;
int g_iNTaps = 1;               /* 1 if no PFB, NUM_TAPS if PFB */
int g_iFileCoeff = 0;
char g_acFileCoeff[256] = {0};
float *g_pfPFBCoeff = NULL;
float *g_pfOptPFBCoeff = NULL;

float g_afConvLookup[256] = {0};

int g_iFileData = 0;
char g_acFileData[256] = {0};

/* PGPLOT global */
float *g_pfFreq = NULL;
float g_fFSamp = 1.0;           /* 1 [frequency] */

int main(int argc, char *argv[])
{
    int iRet = GUPPI_OK;
    int i = 0;
    int iTime = 0;
    int iAcc = DEF_ACC;
    struct timeval stStart = {0};
    struct timeval stStop = {0};
    #if OUTFILE
    int iFile = 0;
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

    #if OUTFILE
    iFile = open("spec.dat",
                 O_CREAT | O_TRUNC | O_WRONLY,
                 S_IRUSR | S_IWUSR | S_IRGRP | S_IROTH);
    if (EXIT_FAILURE == iFile)
    {
        perror("ERROR");
        return EXIT_FAILURE;
    }
    #endif

    (void) gettimeofday(&stStart, NULL);
    while (IsRunning())
    {
        if (g_iIsPFBOn)
        {
            /* do pfb  */
            (void) DoPFB();
        }
        else
        {
            (void) CopyDataForFFT();
        }

        /* do fft - later to GPU*/
        (void) DoFFT();

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
                ++iTime;
                if (iTime == iAcc)
                {
                    #if PLOT
                    /* NOTE: Plot() will modify data! */
                    Plot();
                    usleep(500000);
                    #endif

                    /* dump to buffer */
                    #if OUTFILE
                    for (i = 0; i < g_iNFFT; ++i)
                    {
                        //printf("%d\n", *((int *) &g_pfSumPowX[i]));
                        g_aiSumPowX[i] = *((int *) &g_pfSumPowX[i]);
                        g_aiSumPowY[i] = *((int *) &g_pfSumPowY[i]);
                        g_aiSumStokesRe[i] = *((int *) &g_pfSumStokesRe[i]);
                        g_aiSumStokesIm[i] = *((int *) &g_pfSumStokesIm[i]);
                    }
                    (void) write(iFile, g_aiSumPowX, g_iNFFT * sizeof(int));
                    (void) write(iFile, g_aiSumPowY, g_iNFFT * sizeof(int));
                    (void) write(iFile, g_aiSumStokesRe, g_iNFFT * sizeof(int));
                    (void) write(iFile, g_aiSumStokesIm, g_iNFFT * sizeof(int));
                    #endif

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

    #if OUTFILE
    (void) close(iFile);
    #endif

    CleanUp();

    return GUPPI_OK;
}

/* function that creates the FFT plan, allocates memory, initialises counters,
   etc. */
int Init()
{
    int i = 0;
    int j = 0;
    int k = 0;
    int iRet = GUPPI_OK;

    iRet = RegisterSignalHandlers();
    if (iRet != GUPPI_OK)
    {
        (void) fprintf(stderr, "ERROR: Signal-handler registration failed!\n");
        return GUPPI_ERR_GEN;
    }

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
		g_pfOptPFBCoeff = (float *) malloc(2 * g_iNTaps * g_iNFFT * sizeof(float)); 
        if (NULL == g_pfOptPFBCoeff)
        {
            (void) fprintf(stderr,
                           "ERROR: Memory allocation failed! %s.\n",
                           strerror(errno));
            return GUPPI_ERR_GEN;
        }

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
        else if (iRet != (g_iNTaps * g_iNFFT * sizeof(float)))
        {
            (void) printf("File read done!\n");
        }
        (void) close(g_iFileCoeff);

    	/* duplicate the coefficients for PFB optimisation */
        for (i = 0; i < (g_iNTaps * g_iNFFT); ++i)
        {
            g_pfOptPFBCoeff[2*i] = g_pfPFBCoeff[i];
            g_pfOptPFBCoeff[(2*i)+1] = g_pfPFBCoeff[i];
    	}
    }

    /* allocate memory for data array contents */
    for (i = 0; i < g_iNTaps; ++i)
    {
        g_astPFBData[i].pabDataX = (signed char(*) [][2]) malloc(2 * g_iNFFT * sizeof(signed char));
        if (NULL == g_astPFBData[i].pabDataX)
        {
            (void) fprintf(stderr,
                           "ERROR: Memory allocation failed! %s.\n",
                           strerror(errno));
            return GUPPI_ERR_GEN;
        }
        g_astPFBData[i].pabDataY = (signed char(*) [][2]) malloc(2 * g_iNFFT * sizeof(signed char));
        if (NULL == g_astPFBData[i].pabDataY)
        {
            (void) fprintf(stderr,
                           "ERROR: Memory allocation failed! %s.\n",
                           strerror(errno));
            return GUPPI_ERR_GEN;
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

    for (i = 0; i < g_iNTaps; ++i)
    {
        g_astPFBData[i].pbData = g_pbInBuf + (i * LEN_DATA);
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
            (*g_astPFBData[i].pabDataX)[j][0] = g_astPFBData[i].pbData[k];
            (*g_astPFBData[i].pabDataX)[j][1] = g_astPFBData[i].pbData[k+1];
            (*g_astPFBData[i].pabDataY)[j][0] = g_astPFBData[i].pbData[k+2];
            (*g_astPFBData[i].pabDataY)[j][1] = g_astPFBData[i].pbData[k+3];
            ++j;
        }
    }

    g_iPFBWriteIdx = 0;     /* next write into the first buffer */
    g_iPFBReadIdx = 0;      /* PFB to be performed from first buffer */

    g_pfcFFTInX = (fftwf_complex *) fftwf_malloc(g_iNFFT * sizeof(fftwf_complex));
    if (NULL == g_pfcFFTInX)
    {
        (void) fprintf(stderr,
                       "ERROR: Memory allocation failed! %s.\n",
                       strerror(errno));
        return GUPPI_ERR_GEN;
    }
    g_pfcFFTInY = (fftwf_complex *) fftwf_malloc(g_iNFFT * sizeof(fftwf_complex));
    if (NULL == g_pfcFFTInY)
    {
        (void) fprintf(stderr,
                       "ERROR: Memory allocation failed! %s.\n",
                       strerror(errno));
        return GUPPI_ERR_GEN;
    }
    g_pfcFFTOutX = (fftwf_complex *) fftwf_malloc(g_iNFFT * sizeof(fftwf_complex));
    if (NULL == g_pfcFFTOutX)
    {
        (void) fprintf(stderr,
                       "ERROR: Memory allocation failed! %s.\n",
                       strerror(errno));
        return GUPPI_ERR_GEN;
    }
    g_pfcFFTOutY = (fftwf_complex *) fftwf_malloc(g_iNFFT * sizeof(fftwf_complex));
    if (NULL == g_pfcFFTOutY)
    {
        (void) fprintf(stderr,
                       "ERROR: Memory allocation failed! %s.\n",
                       strerror(errno));
        return GUPPI_ERR_GEN;
    }

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
    int i = 0;
    int j = 0;

    /* write new data to the write buffer */
    g_astPFBData[g_iPFBWriteIdx].pbData += (g_iNTaps * LEN_DATA);
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
    for (i = 0; i < LEN_DATA; i += NUM_BYTES_PER_SAMP)
    {
        (*g_astPFBData[g_iPFBWriteIdx].pabDataX)[j][0] = g_astPFBData[g_iPFBWriteIdx].pbData[i];
        (*g_astPFBData[g_iPFBWriteIdx].pabDataX)[j][1] = g_astPFBData[g_iPFBWriteIdx].pbData[i+1];
        (*g_astPFBData[g_iPFBWriteIdx].pabDataY)[j][0] = g_astPFBData[g_iPFBWriteIdx].pbData[i+2];
        (*g_astPFBData[g_iPFBWriteIdx].pabDataY)[j][1] = g_astPFBData[g_iPFBWriteIdx].pbData[i+3];
        ++j;
    }

    g_iPFBWriteIdx = g_astPFBData[g_iPFBWriteIdx].iNextIdx;
    g_iPFBReadIdx = g_astPFBData[g_iPFBReadIdx].iNextIdx;

    return GUPPI_OK;
}

/* function that performs the PFB */
int DoPFB()
{
    int i = 0;
    int j = 0;
    int k = 0;
    int iCoeffStartIdx = 0;
	signed char *pbIn = NULL;
	float *pfCoeff = NULL;
	float *pfOut = NULL;

    /* reset memory */
    (void) memset(g_pfcFFTInX, '\0', g_iNFFT * sizeof(fftwf_complex));
    (void) memset(g_pfcFFTInY, '\0', g_iNFFT * sizeof(fftwf_complex));

    i = g_iPFBReadIdx;
    for (j = 0; j < g_iNTaps; ++j)
    {
        iCoeffStartIdx = 2 * j * g_iNFFT;
		pfOut = &g_pfcFFTInX[0][0];
		pbIn = &((*g_astPFBData[i].pabDataX)[0][0]);
		pfCoeff = &g_pfOptPFBCoeff[iCoeffStartIdx];
        for (k = 0; k < (2 * g_iNFFT); ++k)
        {
            pfOut[k] += pbIn[k] * pfCoeff[k];
        }
		pfOut = &g_pfcFFTInY[0][0];
		pbIn = &((*g_astPFBData[i].pabDataY)[0][0]);
		pfCoeff = &g_pfOptPFBCoeff[iCoeffStartIdx];
        for (k = 0; k < (2 * g_iNFFT); ++k)
		{
	  		pfOut[k] += pbIn[k] * pfCoeff[k];
		}
        i = g_astPFBData[i].iNextIdx;
    }

    return GUPPI_OK;
}

int CopyDataForFFT()
{
    int i = 0;

    for (i = 0; i < g_iNFFT; ++i)
    {
        g_pfcFFTInX[i][0] = g_afConvLookup[(BYTE) (*g_astPFBData[g_iPFBReadIdx].pabDataX)[i][0]];
        g_pfcFFTInX[i][1] = g_afConvLookup[(BYTE) (*g_astPFBData[g_iPFBReadIdx].pabDataX)[i][1]];
        g_pfcFFTInY[i][0] = g_afConvLookup[(BYTE) (*g_astPFBData[g_iPFBReadIdx].pabDataY)[i][0]];
        g_pfcFFTInY[i][1] = g_afConvLookup[(BYTE) (*g_astPFBData[g_iPFBReadIdx].pabDataY)[i][1]];
    }

    return GUPPI_OK;
}

/* function that performs the FFT */
int DoFFT()
{
    /* execute plan */
    fftwf_execute(g_stPlanX);
    fftwf_execute(g_stPlanY);

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
    free(g_pbInBuf);

    for (i = 0; i < g_iNTaps; ++i)
    {
        fftwf_free(g_astPFBData[i].pabDataX);
        fftwf_free(g_astPFBData[i].pabDataY);
    }
    fftwf_free(g_pfcFFTInX);
    fftwf_free(g_pfcFFTInY);
    fftwf_free(g_pfcFFTOutX);
    fftwf_free(g_pfcFFTOutY);

    free(g_pfPFBCoeff);
	free(g_pfOptPFBCoeff);

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
    for (i = 0; i < g_iNFFT; ++i)
    {
        g_pfSumPowX[i] -= fMaxY;
        printf("%g\n", g_pfSumPowX[i]);
    }
    fMinY -= fMaxY;
    fMaxY = 0;
    //printf("********************************\n");
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

