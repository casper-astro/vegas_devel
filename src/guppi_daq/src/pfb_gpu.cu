#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>
//#include <cuda.h>
#include <cufft.h>

#include "guppi_error.h"
#include "guppi_defines.h"
#include "guppi_databuf.h"
#include "pfb_gpu.h"
#include "pfb_gpu_kernels.h"
#include "spead_heap.h"

extern int run;

/**
 * Global variables: maybe move this to a struct that is passed to each function?
 */
size_t buf_in_block_size;
size_t buf_out_block_size;
size_t buf_index_size;
int nchan;

struct databuf_index *device_in_index;
struct databuf_index *device_out_index;

int g_iMaxThreadsPerBlock = 0;
cufftHandle g_stPlan = {0};
float4* g_pf4FFTIn_d = NULL;
float4* g_pf4FFTOut_d = NULL;
char4* g_pc4InBuf = NULL;
char4* g_pc4InBufRead = NULL;
char4* g_pc4Data_d = NULL;              /* raw data starting address */
char4* g_pc4DataRead_d = NULL;          /* raw data read pointer */
dim3 g_dimBPFB(1, 1, 1);
dim3 g_dimGPFB(1, 1);
dim3 g_dimBAccum(1, 1, 1);
dim3 g_dimGAccum(1, 1);
dim3 g_dimBConv(1, 1, 1);
dim3 g_dimGConv(1, 1);
float4* g_pf4SumStokes_d = NULL;
int4 *g_pi4SumStokes_d = NULL;
int g_iNumSubBands = 0;
int g_iTime = 0;
int g_iAccLen = 4;
unsigned int g_iPrevStatusBits = 0;

void __CUDASafeCallWithCleanUp(cudaError_t iCUDARet,
                               const char* pcFile,
                               const int iLine,
                               void (*pCleanUp)(void));

#define CUDASafeCallWithCleanUp(iRet)   __CUDASafeCallWithCleanUp(iRet,       \
                                                                  __FILE__,   \
                                                                  __LINE__,   \
                                                                  &cleanup_gpu)

/* Initialize all necessary memory, etc for doing PFB 
 * at the given params.
 */
extern "C"
void init_gpu(size_t input_block_sz, size_t output_block_sz, size_t index_sz, int num_subbands, int num_chans)
{
    int iDevCount = 0;
    cudaDeviceProp stDevProp = {0};
    cufftResult iCUFFTRet = CUFFT_SUCCESS;

	buf_in_block_size = input_block_sz;
    buf_out_block_size = output_block_sz;
	buf_index_size = index_sz;
    nchan = num_chans;
    /* set the number of sub-bands */
    g_iNumSubBands = num_subbands;

    /* since CUDASafeCallWithCleanUp() calls cudaGetErrorString(),
       it should not be used here - will cause crash if no CUDA device is
       found */
    (void) cudaGetDeviceCount(&iDevCount);
    if (0 == iDevCount)
    {
        (void) fprintf(stderr, "ERROR: No CUDA-capable device found!\n");
        run = 0;
        return;
    }

    /* just use the first device */
    CUDASafeCallWithCleanUp(cudaSetDevice(0));

    CUDASafeCallWithCleanUp(cudaGetDeviceProperties(&stDevProp, 0));
    g_iMaxThreadsPerBlock = stDevProp.maxThreadsPerBlock;

    /* allocate memory for data array - 32MB is the block size for the VEGAS
       input buffer, allocate 32MB + space for (VEGAS_NUM_TAPS - 1) blocks of
       data */
    CUDASafeCallWithCleanUp(cudaMalloc((void **) &g_pc4Data_d,
                                       (buf_in_block_size
                                        + ((VEGAS_NUM_TAPS - 1)
                                           * g_iNumSubBands
                                           * nchan
                                           * sizeof(char4)))));
    g_pc4DataRead_d = g_pc4Data_d;

    /* calculate kernel parameters */
    /* ASSUMPTION: nchan >= g_iMaxThreadsPerBlock */
    g_dimBPFB.x = g_iMaxThreadsPerBlock;
    g_dimBAccum.x = g_iMaxThreadsPerBlock;
    g_dimGPFB.x = (g_iNumSubBands * nchan) / g_iMaxThreadsPerBlock;
    g_dimGAccum.x = (g_iNumSubBands * nchan) / g_iMaxThreadsPerBlock;

    g_dimBConv.x = g_iMaxThreadsPerBlock;
    g_dimGConv.x = buf_out_block_size / (sizeof(int4) * g_iMaxThreadsPerBlock);
    printf("******************griddim = %d\n", g_dimGConv.x);

    CUDASafeCallWithCleanUp(cudaMalloc((void **) &g_pf4FFTIn_d,
                                 g_iNumSubBands * nchan * sizeof(float4)));
    CUDASafeCallWithCleanUp(cudaMalloc((void **) &g_pf4FFTOut_d,
                                 g_iNumSubBands * nchan * sizeof(float4)));
    CUDASafeCallWithCleanUp(cudaMalloc((void **) &g_pf4SumStokes_d,
                                 g_iNumSubBands * nchan * sizeof(float4)));
    CUDASafeCallWithCleanUp(cudaMemset(g_pf4SumStokes_d,
                                 '\0',
                                 g_iNumSubBands * nchan * sizeof(float4)));
    CUDASafeCallWithCleanUp(cudaMalloc((void **) &g_pi4SumStokes_d,
                                 g_iNumSubBands * nchan * sizeof(int4)));

    /* create plan */
    iCUFFTRet = cufftPlanMany(&g_stPlan,
                              FFTPLAN_RANK,
                              &nchan,
                              &nchan,
                              FFTPLAN_ISTRIDE,
                              FFTPLAN_IDIST,
                              &nchan,
                              FFTPLAN_OSTRIDE,
                              FFTPLAN_ODIST,
                              CUFFT_C2C,
                              FFTPLAN_BATCH);
    if (iCUFFTRet != CUFFT_SUCCESS)
    {
        (void) fprintf(stderr, "ERROR: Plan creation failed!\n");
        run = 0;
        return;
    }

    return;
}


/* Actually do the PFB by calling CUDA kernels */
extern "C"
void do_pfb(char *in, char *out, struct databuf_index *index_in,
                struct databuf_index *index_out, int first)
{
    /* Declare local variables */
    int heap_in = 0;
    int heap_out = 0;
    char *heap_addr_in, *heap_addr_out;
    struct time_spead_heap* time_heap_in;
    struct freq_spead_heap* freq_heap_out;
    int iProcData = 0;
    cudaError_t iCUDARet = cudaSuccess;
    int iRet = GUPPI_OK;

    /* Set basic params in output index */
    //index_out->num_heaps = index_in->num_heaps;
    index_out->num_heaps = buf_in_block_size / (sizeof(char4) * g_iNumSubBands * nchan * g_iAccLen);
    printf("*************Num out heaps = %d\n", index_out->num_heaps);
    //index_out->heap_size = sizeof(struct freq_spead_heap) + nchan * 4 * 4;
    index_out->heap_size = sizeof(struct freq_spead_heap) + (g_iNumSubBands * nchan * sizeof(int4));
    printf("*************Out heap size = %d\n", index_out->heap_size);

    /* Write new SPEAD header fields for output heap */
    /* Note: this cannot be done on GPU, due to alignment problems */
    //for (heap = 0; heap < index_in->num_heaps; heap++)
    //{
    //}

    /* Copy data block to GPU */
    if (first)
    {
        CUDASafeCallWithCleanUp(cudaMemcpy(g_pc4Data_d,
                                           in,
                                           buf_in_block_size + ((VEGAS_NUM_TAPS - 1) * g_iNumSubBands * nchan * sizeof(char4)),
                                           cudaMemcpyHostToDevice));
    }
    else
    {
        CUDASafeCallWithCleanUp(cudaMemcpy(g_pc4Data_d,
                                           g_pc4Data_d + (buf_in_block_size / sizeof(char4)),
                                           ((VEGAS_NUM_TAPS - 1) * g_iNumSubBands * nchan * sizeof(char4)),
                                           cudaMemcpyDeviceToDevice));
        CUDASafeCallWithCleanUp(cudaMemcpy(g_pc4Data_d + ((VEGAS_NUM_TAPS - 1) * g_iNumSubBands * nchan),
                                           in,
                                           buf_in_block_size,
                                           cudaMemcpyHostToDevice));
    }

    /* calculate input heap addresse */
    heap_in = 0;
    heap_addr_in = in + index_in->heap_size*heap_in;
    time_heap_in = (struct time_spead_heap*)(heap_addr_in);

    g_pc4DataRead_d = g_pc4Data_d;
    iProcData = 0;
    while (buf_in_block_size != iProcData)  /* loop till 32MB of data is processed */
    {
        /* perform polyphase filtering */
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
            run = 0;
            break;
        }
         
        iRet = do_fft();
        if (iRet != GUPPI_OK)
        {
            (void) fprintf(stderr, "ERROR: FFT failed!\n");
            run = 0;
            break;
        }

        /* accumulate power x, power y, stokes real and imag, if the blanking
           bit is not set */
        if ((time_heap_in->status_bits & 0x08) == 0)
        {
            iRet = accumulate();
            if (iRet != GUPPI_OK)
            {
                (void) fprintf(stderr, "ERROR: Accumulation failed!\n");
                run = 0;
                break;
            }
            ++g_iTime;
        }
        else
        {
            /* state just changed */
            if ((g_iPrevStatusBits & 0x08) == 1)
            {
                /* dump to buffer */
                heap_addr_out = out + index_out->heap_size*heap_out;
                freq_heap_out = (struct freq_spead_heap*)(heap_addr_out);
         
                /* Write new heap header fields */
                freq_heap_out->time_cntr_id = 0x20;
                freq_heap_out->time_cntr = time_heap_in->time_cntr;
                freq_heap_out->spectrum_cntr_id = 0x21;
                freq_heap_out->spectrum_cntr = index_in->cpu_gpu_buf[heap_out].heap_cntr;
                freq_heap_out->integ_size_id = 0x22;
                freq_heap_out->integ_size = 32;
                freq_heap_out->mode_id = 0x23;
                freq_heap_out->mode = time_heap_in->mode;
                freq_heap_out->status_bits_id = 0x24;
                freq_heap_out->status_bits = time_heap_in->status_bits;
                freq_heap_out->payload_data_off_addr_mode = 0x80;
                freq_heap_out->payload_data_off_id = 0x25;
                freq_heap_out->payload_data_off = 0;

                /* Update output index */
                index_out->cpu_gpu_buf[heap_out].heap_valid =
                         index_in->cpu_gpu_buf[heap_out].heap_valid;
                index_out->cpu_gpu_buf[heap_out].heap_cntr =
                        index_in->cpu_gpu_buf[heap_out].heap_cntr;
                index_out->cpu_gpu_buf[heap_out].heap_rcvd_mjd =
                         index_in->cpu_gpu_buf[heap_out].heap_rcvd_mjd ;

                iRet = get_accumulated_spectrum_from_device(heap_addr_out + sizeof(struct freq_spead_heap));
                if (iRet != GUPPI_OK)
                {
                    (void) fprintf(stderr, "ERROR: Getting accumulated spectrum failed!\n");
                    run = 0;
                    break;
                }

                //printf("Wrote to heap_out = %d\n", heap_out);
                ++heap_out;

                /* zero accumulators */
                zero_accumulator();
            }
            /* reset time */
            g_iTime = 0;
        }
        g_iPrevStatusBits = time_heap_in->status_bits;

        if (g_iTime == g_iAccLen)
        {
            /* dump to buffer */
            heap_addr_out = out + index_out->heap_size*heap_out;
            freq_heap_out = (struct freq_spead_heap*)(heap_addr_out);
     
            /* Write new heap header fields */
            freq_heap_out->time_cntr_id = 0x20;
            freq_heap_out->time_cntr = time_heap_in->time_cntr;//???
            freq_heap_out->spectrum_cntr_id = 0x21;
            freq_heap_out->spectrum_cntr = index_in->cpu_gpu_buf[heap_out].heap_cntr;//???
            freq_heap_out->integ_size_id = 0x22;
            freq_heap_out->integ_size = 32;
            freq_heap_out->mode_id = 0x23;
            freq_heap_out->mode = time_heap_in->mode;
            freq_heap_out->status_bits_id = 0x24;
            freq_heap_out->status_bits = time_heap_in->status_bits;
            freq_heap_out->payload_data_off_addr_mode = 0x80;
            freq_heap_out->payload_data_off_id = 0x25;
            freq_heap_out->payload_data_off = 0;

            /* Update output index */
            index_out->cpu_gpu_buf[heap_out].heap_valid =
                     index_in->cpu_gpu_buf[heap_out].heap_valid;
            index_out->cpu_gpu_buf[heap_out].heap_cntr =
                    index_in->cpu_gpu_buf[heap_out].heap_cntr;

            iRet = get_accumulated_spectrum_from_device(heap_addr_out + sizeof(struct freq_spead_heap));
            if (iRet != GUPPI_OK)
            {
                (void) fprintf(stderr, "ERROR: Getting accumulated spectrum failed!\n");
                run = 0;
                break;
            }

            printf("Wrote to heap_out = %d\n", heap_out);
            ++heap_out;

            /* zero accumulators */
            zero_accumulator();
            /* reset time */
            g_iTime = 0;
        }

        iProcData += (g_iNumSubBands * nchan * sizeof(char4));
        /* update the data read pointer */
        g_pc4DataRead_d += (g_iNumSubBands * nchan);

        /* calculate input heap addresses to read headers */
        heap_addr_in = in + index_in->heap_size*heap_in;
        time_heap_in = (struct time_spead_heap*)(heap_addr_in);

        /* TODO: check all heaps in this PFB, if any is blanked, no accumulation */
    }

    return;
}

/* function that performs the FFT */
int do_fft()
{
    cufftResult iCUFFTRet = CUFFT_SUCCESS;

    /* execute plan */
    iCUFFTRet = cufftExecC2C(g_stPlan,
                             (cufftComplex*) g_pf4FFTIn_d,
                             (cufftComplex*) g_pf4FFTOut_d,
                             CUFFT_FORWARD);
    if (iCUFFTRet != CUFFT_SUCCESS)
    {
        (void) fprintf(stderr, "ERROR: FFT failed!");
        run = 0;
        return GUPPI_ERR_GEN;
    }

    return GUPPI_OK;
}

int accumulate()
{
    cudaError_t iCUDARet = cudaSuccess;

    Accumulate<<<g_dimGAccum, g_dimBAccum>>>(g_pf4FFTOut_d,
                                             g_pf4SumStokes_d,
                                             g_pi4SumStokes_d);
    CUDASafeCallWithCleanUp(cudaThreadSynchronize());
    iCUDARet = cudaGetLastError();
    if (iCUDARet != cudaSuccess)
    {
        (void) fprintf(stderr, cudaGetErrorString(iCUDARet));
        run = 0;
        return GUPPI_ERR_GEN;
    }

    return GUPPI_OK;
}

void zero_accumulator()
{
    CUDASafeCallWithCleanUp(cudaMemset(g_pf4SumStokes_d,
                                 '\0',
                                 (g_iNumSubBands
                                  * nchan
                                  * sizeof(float4))));

    return;
}

int get_accumulated_spectrum_from_device(char *out)
{
    cudaError_t iCUDARet = cudaSuccess;

    //Convert<<<g_dimGConv, g_dimBConv>>>(g_pf4SumStokes_d,
    //                                    g_pi4SumStokes_d);
    //CUDASafeCallWithCleanUp(cudaThreadSynchronize());
    //iCUDARet = cudaGetLastError();
    if (iCUDARet != cudaSuccess)
    {
        (void) fprintf(stderr, cudaGetErrorString(iCUDARet));
        run = 0;
        return GUPPI_ERR_GEN;
    }
    CUDASafeCallWithCleanUp(cudaMemcpy(out,
                                       g_pi4SumStokes_d,
                                       (g_iNumSubBands
                                        * nchan
                                        * sizeof(int4)),
                                       cudaMemcpyDeviceToHost));

    return GUPPI_OK;
}

void __CUDASafeCallWithCleanUp(cudaError_t iCUDARet,
                               const char* pcFile,
                               const int iLine,
                               void (*pCleanUp)(void))
{
    if (iCUDARet != cudaSuccess)
    {
        (void) fprintf(stderr,
                       "ERROR: File <%s>, Line %d: %s\n",
                       pcFile,
                       iLine,
                       cudaGetErrorString(iCUDARet));
        run = 0;
        return;
    }

    return;
}

/* 
 * Frees up any allocated memory.
 */
void cleanup_gpu()
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
    if (g_pf4SumStokes_d != NULL)
    {
        (void) cudaFree(g_pf4SumStokes_d);
        g_pf4SumStokes_d = NULL;
    }
    if (g_pi4SumStokes_d != NULL)
    {
        (void) cudaFree(g_pi4SumStokes_d);
        g_pi4SumStokes_d = NULL;
    }

    /* destroy plan */
    (void) cufftDestroy(g_stPlan);

    return;
}

