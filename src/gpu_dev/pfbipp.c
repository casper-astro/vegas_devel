/* 
 * gsp3mcpu.c
 * GBT Spectrometer Priority 3 Mode - Stand-Alone CPU Implementation
 *
 * Created by Jayanth Chennamangalam on 2011.06.02
 */

#include <ipps.h>
#include "gsp3mcpu.h"

#define PLOT    0
#define MEM     1

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

int g_iIsPFBOn = DEF_PFB_ON;
int g_iNTaps = 1;               /* 1 if no PFB, NUM_TAPS if PFB */
int g_iFileCoeff = 0;
char g_acFileCoeff[256] = {0};
float *g_pfPFBCoeff = NULL;
float *AltCoeff = NULL;

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
    int iStart = 0;
    int iStop = 0;
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

    printf("Size of fftw_complex is %d bytes\n",(int)sizeof(fftwf_complex));

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

    iStart = clock();
    while (IsRunning())
    {
        if (g_iIsPFBOn)
        {
            /* do pfb - later to GPU */
	  (void) DoPFB(0);
	  //(void) DoPFBipp();
	  //(void) DoPFBptr();
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
                /* TODO: when blanking is unset, clear memory and start over */
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
    iStop = clock();
    (void) printf("Time taken: %gs\n", ((float) (iStop - iStart)) / CLOCKS_PER_SEC);

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
	AltCoeff = (float *) malloc(g_iNTaps * g_iNFFT * sizeof(float)*2); 
        if (NULL == AltCoeff)
        {
            (void) fprintf(stderr,
                           "ERROR: Memory allocation failed! %s.\n",
                           strerror(errno));
            return GUPPI_ERR_GEN;
        }

        if (NULL == g_pfPFBCoeff)
        {
            (void) fprintf(stderr,
                           "ERROR: Memory allocation failed! %s.\n",
                           strerror(errno));
            return GUPPI_ERR_GEN;
        }

        /* read filter coefficients */
        (void) strncpy(g_acFileCoeff, FILE_COEFF, 256);
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
    }
    // Duplicate PFB Coefficients
    for (k=0;k<g_iNTaps*g_iNFFT;k++){
      AltCoeff[2*k]=AltCoeff[2*k+1]=g_pfPFBCoeff[k];
    }

    /* allocate memory for the PFB_DATA array contents */
    for (i = 0; i < g_iNTaps; ++i)
    {
        #if MEM
        #else
        g_astPFBData[i].pbData = (BYTE *) malloc(LEN_DATA * sizeof(BYTE));
        if (NULL == g_astPFBData[i].pbData)
        {
            (void) fprintf(stderr,
                           "ERROR: Memory allocation failed! %s.\n",
                           strerror(errno));
            return GUPPI_ERR_GEN;
        }
        #endif
        g_astPFBData[i].pfData = (float *) malloc(LEN_DATA * sizeof(float));
        if (NULL == g_astPFBData[i].pfData)
        {
            (void) fprintf(stderr,
                           "ERROR: Memory allocation failed! %s.\n",
                           strerror(errno));
            return GUPPI_ERR_GEN;
        }

        g_astPFBData[i].pfcDataX = (fftwf_complex *) fftwf_malloc(g_iNFFT * sizeof(fftwf_complex));
        if (NULL == g_astPFBData[i].pfcDataX)
        {
            (void) fprintf(stderr,
                           "ERROR: Memory allocation failed! %s.\n",
                           strerror(errno));
            return GUPPI_ERR_GEN;
        }
        g_astPFBData[i].pfcDataY = (fftwf_complex *) fftwf_malloc(g_iNFFT * sizeof(fftwf_complex));
        if (NULL == g_astPFBData[i].pfcDataY)
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

    #if MEM
    /* load data into memory */
    iRet = LoadData();
    if (iRet != GUPPI_OK)
    {
        (void) fprintf(stderr,
                       "ERROR! Data loading failed!\n");
        return GUPPI_ERR_GEN;
    }
    #else
    #endif

    for (i = 0; i < g_iNTaps; ++i)
    {
        #if MEM
        g_astPFBData[i].pbData = g_pbInBuf + (i * LEN_DATA);
        ++g_iReadCount;
        if (g_iReadCount == g_iNumReads)
        {
            (void) printf("Data read done!\n");
            g_iIsDone = TRUE;
        }
        #else
        iRet = read(g_iFileData, g_astPFBData[i].pbData, LEN_DATA);
        if (GUPPI_ERR_GEN == iRet)
        {
            (void) fprintf(stderr,
                           "ERROR: Data reading failed! %s.\n",
                           strerror(errno));
            return GUPPI_ERR_GEN;
        }
        else if (iRet != LEN_DATA)
        {
            (void) printf("File read done!\n");
            g_iIsDone = TRUE;
        }
        #endif

        Convert(g_astPFBData[i].pbData, g_astPFBData[i].pfData);

        /* unpack data */
        /* assuming real and imaginary parts are interleaved, and X and Y are
           interleaved, like so:
           reX, imX, reY, imY, ... */
        j = 0;
        for (k = 0; k < LEN_DATA; k += NUM_BYTES_PER_SAMP)
        {
            g_astPFBData[i].pfcDataX[j][0] = g_astPFBData[i].pfData[k];
            g_astPFBData[i].pfcDataX[j][1] = g_astPFBData[i].pfData[k+1];
            g_astPFBData[i].pfcDataY[j][0] = g_astPFBData[i].pfData[k+2];
            g_astPFBData[i].pfcDataY[j][1] = g_astPFBData[i].pfData[k+3];
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

    /* just for plotting */
    InitPlot();

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
    #if MEM
    #else
    int iRet = GUPPI_OK;
    #endif
    int i = 0;
    int j = 0;

    /* write new data to the write buffer */
    #if MEM
    g_astPFBData[g_iPFBWriteIdx].pbData += (g_iNTaps * LEN_DATA);
    ++g_iReadCount;
    if (g_iReadCount == g_iNumReads)
    {
        (void) printf("Data read done!\n");
        g_iIsDone = TRUE;
    }
    #else
    iRet = read(g_iFileData, g_astPFBData[g_iPFBWriteIdx].pbData, LEN_DATA);
    if (GUPPI_ERR_GEN == iRet)
    {
        (void) fprintf(stderr,
                       "ERROR: Data reading failed! %s.\n",
                       strerror(errno));
        return GUPPI_ERR_GEN;
    }
    else if (iRet != LEN_DATA)
    {
        (void) printf("File read done!\n");
        g_iIsDone = TRUE;
    }
    #endif

    /* convert data format */
    Convert(g_astPFBData[g_iPFBWriteIdx].pbData, g_astPFBData[g_iPFBWriteIdx].pfData);

    /* unpack data */
    /* assuming real and imaginary parts are interleaved, and X and Y are
       interleaved, like so:
       reX, imX, reY, imY, ... */
    j = 0;
    for (i = 0; i < LEN_DATA; i += NUM_BYTES_PER_SAMP)
    {
        g_astPFBData[g_iPFBWriteIdx].pfcDataX[j][0] = g_astPFBData[g_iPFBWriteIdx].pfData[i];
        g_astPFBData[g_iPFBWriteIdx].pfcDataX[j][1] = g_astPFBData[g_iPFBWriteIdx].pfData[i+1];
        g_astPFBData[g_iPFBWriteIdx].pfcDataY[j][0] = g_astPFBData[g_iPFBWriteIdx].pfData[i+2];
        g_astPFBData[g_iPFBWriteIdx].pfcDataY[j][1] = g_astPFBData[g_iPFBWriteIdx].pfData[i+3];
        ++j;
    }

    g_iPFBWriteIdx = g_astPFBData[g_iPFBWriteIdx].iNextIdx;
    g_iPFBReadIdx = g_astPFBData[g_iPFBReadIdx].iNextIdx;

    return GUPPI_OK;
}

/* function that converts Fix_8_7 format data to floating-point */
void Convert(BYTE *pcData, float *pfData)
{
    int i = 0;

    for (i = 0; i < LEN_DATA; ++i)
    {
        pfData[i] = g_afConvLookup[pcData[i]];
    }

    return;
}

/* function that performs the PFB */
int DoPFB(int ipp)
{
    int i = 0;
    int j = 0;
    int k = 0;
    int iCoeffStartIdx = 0;

    /* reset memory */
    (void) memset(g_pfcFFTInX, '\0', g_iNFFT * sizeof(fftwf_complex));
    (void) memset(g_pfcFFTInY, '\0', g_iNFFT * sizeof(fftwf_complex));

    i = g_iPFBReadIdx;
    while (j < g_iNTaps)
    {
        iCoeffStartIdx = j * g_iNFFT;
	float * out = &g_pfcFFTInX[0][0];
	float * in = &g_astPFBData[i].pfcDataX[0][0];
	float * co = &AltCoeff[2*iCoeffStartIdx];
	if (ipp){
	  //ippsAddProduct_32f(in,co,out,g_iNFFT*2);
	}else{
	  for (k = 0; k < g_iNFFT*2; ++k)
	      out[k]+=in[k]*co[k];
	}
	out = &g_pfcFFTInY[0][0];
	in = &g_astPFBData[i].pfcDataY[0][0];
	co = &AltCoeff[2*iCoeffStartIdx];
	if (ipp){
	  //ippsAddProduct_32f(in,co,out,g_iNFFT*2);
	}
	else{
        for (k = 0; k < g_iNFFT*2; ++k)
	  out[k]+=in[k]*co[k];
	}
        i = g_astPFBData[i].iNextIdx;
        ++j;
    }

    return GUPPI_OK;
}

int DoPFBptr()
{
  float * x0, *x1, *y0, *y1;
  float * ax0, *ax1, *ay0, *ay1;
  float * fc;

    int i = 0;
    int j = 0;
    int k = 0;
    int iCoeffStartIdx = 0;

    /* reset memory */
    (void) memset(g_pfcFFTInX, '\0', g_iNFFT * sizeof(fftwf_complex));
    (void) memset(g_pfcFFTInY, '\0', g_iNFFT * sizeof(fftwf_complex));

    i = g_iPFBReadIdx;
    while (j < g_iNTaps)
    {
        iCoeffStartIdx = j * g_iNFFT;
	x0 = (float *) & g_astPFBData[i].pfcDataX[0][0];
	x1 = (float *) & g_astPFBData[i].pfcDataX[0][1];
	y0 = (float *) & g_astPFBData[i].pfcDataY[0][0];
	y1 = (float *) & g_astPFBData[i].pfcDataY[0][1];
	fc = (float *) & g_pfPFBCoeff[iCoeffStartIdx];
	ax0 = (float *) & g_pfcFFTInX[0][0];
	ax1 = (float *) & g_pfcFFTInX[0][1];
	ay0 = (float *) & g_pfcFFTInY[0][0];
	ay1 = (float *) & g_pfcFFTInY[0][1];
        for (k = 0; k < g_iNFFT; ++k)
        {
	  *ax0+= *x0* *fc;
	  *ax1+= *x1* *fc;
	  *ay0+= *y0* *fc;
	  *ay1+= *y1* *fc;
	  x0++;x1++;y0++;y1++;
	  fc++;
	  ax0++; ax1++; ay0++;ay1++;
        }
        i = g_astPFBData[i].iNextIdx;
        ++j;
    }

    return GUPPI_OK;
}

/* function that performs the PFB */
int DoPFBipp()
{
    int i = 0;
    int j = 0;
    int k = 0;
    int iCoeffStartIdx = 0;

    /* reset memory */
    (void) memset(g_pfcFFTInX, '\0', g_iNFFT * sizeof(fftwf_complex));
    (void) memset(g_pfcFFTInY, '\0', g_iNFFT * sizeof(fftwf_complex));

    i = g_iPFBReadIdx;
    while (j < g_iNTaps)
    {
        iCoeffStartIdx = j * g_iNFFT;
	/*	ippsAddProduct_32f(&g_astPFBData[i].pfcDataX[0][0], &g_pfPFBCoeff[iCoeffStartIdx],
			   &g_pfcFFTInX[0][0],g_iNFFT);
	ippsAddProduct_32f(&g_astPFBData[i].pfcDataX[0][1], &g_pfPFBCoeff[iCoeffStartIdx],
			   &g_pfcFFTInX[0][1],g_iNFFT);
	ippsAddProduct_32f(&g_astPFBData[i].pfcDataY[0][0], &g_pfPFBCoeff[iCoeffStartIdx],
			   &g_pfcFFTInY[0][0],g_iNFFT);
	ippsAddProduct_32f(&g_astPFBData[i].pfcDataY[0][1], &g_pfPFBCoeff[iCoeffStartIdx],
			   &g_pfcFFTInY[0][1],g_iNFFT);
	*/
	//        for (k = 0; k < g_iNFFT; ++k)
        //{
        //    g_pfcFFTInX[k][0] += g_astPFBData[i].pfcDataX[k][0] * g_pfPFBCoeff[iCoeffStartIdx+k];
        //    g_pfcFFTInX[k][1] += g_astPFBData[i].pfcDataX[k][1] * g_pfPFBCoeff[iCoeffStartIdx+k];
        //    g_pfcFFTInY[k][0] += g_astPFBData[i].pfcDataY[k][0] * g_pfPFBCoeff[iCoeffStartIdx+k];
        //    g_pfcFFTInY[k][1] += g_astPFBData[i].pfcDataY[k][1] * g_pfPFBCoeff[iCoeffStartIdx+k];
        //}
        i = g_astPFBData[i].iNextIdx;
        ++j;
    }

    return GUPPI_OK;
}

int CopyDataForFFT()
{
    (void) memcpy(g_pfcFFTInX, g_astPFBData[g_iPFBReadIdx].pfcDataX, g_iNFFT * sizeof(fftwf_complex));
    (void) memcpy(g_pfcFFTInY, g_astPFBData[g_iPFBReadIdx].pfcDataY, g_iNFFT * sizeof(fftwf_complex));

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
    #if MEM
    #else
        free(g_astPFBData[i].pbData);
    #endif
        free(g_astPFBData[i].pfData);
        fftwf_free(g_astPFBData[i].pfcDataX);
        fftwf_free(g_astPFBData[i].pfcDataY);
    }
    fftwf_free(g_pfcFFTInX);
    fftwf_free(g_pfcFFTInY);
    fftwf_free(g_pfcFFTOutX);
    fftwf_free(g_pfcFFTOutY);

    free(g_pfPFBCoeff);

    free(g_pfSumPowX);
    free(g_pfSumPowY);
    free(g_pfSumStokesRe);
    free(g_pfSumStokesIm);

    /* destroy plans */
    fftwf_destroy_plan(g_stPlanX);
    fftwf_destroy_plan(g_stPlanY);

    fftwf_cleanup();

    (void) close(g_iFileData);

    /* for plotting */
    free(g_pfFreq);
    //cpgclos();

    return;
}

void InitPlot()
{
    int iRet = GUPPI_OK;
    int i = 0;

    iRet = 1;//cpgopen(PG_DEV);
    if (iRet <= 0)
    {
        (void) fprintf(stderr,
                       "ERROR: Opening graphics device %s failed!\n",
                       PG_DEV);
        return;
    }

    //cpgsch(2);
    //cpgsubp(1, 4);

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
    float fMinY = 0.0;
    float fMaxY = 10.0;
    int i = 0;

    /* take log10 of data */
    for (i = 0; i < g_iNFFT; ++i)
    {
        g_pfSumPowX[i] = log10f(g_pfSumPowX[i]);
        g_pfSumPowY[i] = log10f(g_pfSumPowY[i]);
    }

    /* plot g_pfSumPowX */
    //cpgpanl(1, 1);
    //cpgeras();
    //cpgsvp(PG_VP_ML, PG_VP_MR, PG_VP_MB, PG_VP_MT);
    //cpgswin(fMinFreq, fMaxFreq, fMinY, fMaxY);
    //cpglab("Bin Number",
    //         "",
    //     "SumPowX");
    //cpgsci(PG_CI_PLOT);
    //cpgline(g_iNFFT, g_pfFreq, g_pfSumPowX);
    //cpgsci(PG_CI_DEF);
    //cpgbox("BCNST", 0.0, 0, "BCNST", 0.0, 0);

//    fMaxY = 1e4;
    /* plot g_pfSumPowY */
    //cpgpanl(1, 2);
    //cpgeras();
    //cpgsvp(PG_VP_ML, PG_VP_MR, PG_VP_MB, PG_VP_MT);
    //cpgswin(fMinFreq, fMaxFreq, fMinY, fMaxY);
    //cpglab("Bin Number",
//         "",
//         "SumPowY");
    //cpgsci(PG_CI_PLOT);
    //cpgline(g_iNFFT, g_pfFreq, g_pfSumPowY);
    //cpgsci(PG_CI_DEF);
    //cpgbox("BCNST", 0.0, 0, "BCNST", 0.0, 0);

    fMinY = -1e3;
    fMaxY = 1e3;

    /* plot g_pfSumStokesRe */
    //cpgpanl(1, 3);
    //cpgeras();
    //cpgsvp(PG_VP_ML, PG_VP_MR, PG_VP_MB, PG_VP_MT);
    //cpgswin(fMinFreq, fMaxFreq, fMinY, fMaxY);
    //cpglab("Bin Number",
//         "",
//         "SumStokesRe");
    //cpgsci(PG_CI_PLOT);
    //cpgline(g_iNFFT, g_pfFreq, g_pfSumStokesRe);
    //cpgsci(PG_CI_DEF);
    //cpgbox("BCNST", 0.0, 0, "BCNST", 0.0, 0);

    /* plot g_pfSumStokesIm */
    //cpgpanl(1, 4);
    //cpgeras();
    //cpgsvp(PG_VP_ML, PG_VP_MR, PG_VP_MB, PG_VP_MT);
    //cpgswin(fMinFreq, fMaxFreq, fMinY, fMaxY);
    //cpglab("Bin Number",
//  "",
//          "SumStokesIm");
    //cpgsci(PG_CI_PLOT);
    //cpgline(g_iNFFT, g_pfFreq, g_pfSumStokesIm);
    //cpgsci(PG_CI_DEF);
    //cpgbox("BCNST", 0.0, 0, "BCNST", 0.0, 0);

    return;
}

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
    (void) printf("    -s  --fsamp <value>                  ");
    (void) printf("Sampling frequency\n");

    return;
}

