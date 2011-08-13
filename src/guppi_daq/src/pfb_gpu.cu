#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <pthread.h>
#include <errno.h>
#include <cufft.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>

#include "fitshead.h"
#include "guppi_error.h"
#ifdef __cplusplus
extern "C" {
#endif
#include "guppi_status.h"
#include "guppi_databuf.h"
#ifdef __cplusplus
}
#endif
#include "guppi_defines.h"
#include "pfb_gpu.h"
#include "pfb_gpu_kernels.h"
#include "spead_heap.h"

/* TODO: move to .h? */
#define STATUS_KEY "GPUSTAT"

/* ASSUMPTIONS: 1. All blocks contain the same number of heaps. */

extern int run;

/**
 * Global variables: maybe move this to a struct that is passed to each function?
 */
size_t g_buf_in_block_size;
size_t g_buf_out_block_size;
size_t g_buf_index_size;
int g_nchan;

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
float4* g_pf4SumStokes_d = NULL;
int g_num_subbands = 0;
int g_iFileCoeff = 0;
char g_acFileCoeff[256] = {0};
signed char *g_pcPFBCoeff = NULL;
signed char *g_pcPFBCoeff_d = NULL;
unsigned int g_iPrevBlankingState = FALSE;
int g_tot_heap_out = 0;
int g_iMaxNumHeapOut = 0;
int g_pfb_curblock_out = 0;
int g_heap_out = 0;
int g_block_in_data_size = 0;
int g_iTailValid = TRUE;
/* these arrays need to be only a little longer than MAX_HEAPS_PER_BLK, but
   since we don't know the exact length, just allocate twice that value */
unsigned int g_auiStatusBits[2*MAX_HEAPS_PER_BLK] = {0};
unsigned int g_auiHeapValid[2*MAX_HEAPS_PER_BLK] = {0};
int g_first_heap_in = 0;
double g_first_heap_rcvd_mjd = 0.0;
int g_iSpecPerAcc = 0;

void __CUDASafeCall(cudaError_t iCUDARet,
                               const char* pcFile,
                               const int iLine,
                               void (*pCleanUp)(void));

#define CUDASafeCall(iRet)   __CUDASafeCall(iRet,       \
                                                                  __FILE__,   \
                                                                  __LINE__,   \
                                                                  &cleanup_gpu)

/* Initialize all necessary memory, etc for doing PFB 
 * at the given params.
 */
extern "C"
int init_gpu(size_t input_block_sz, size_t output_block_sz, size_t index_sz, int num_subbands, int num_chans)
{
    int iDevCount = 0;
    cudaDeviceProp stDevProp = {0};
    cufftResult iCUFFTRet = CUFFT_SUCCESS;
    int iRet = EXIT_SUCCESS;

    g_buf_in_block_size = input_block_sz;
    g_buf_out_block_size = output_block_sz;
    g_buf_index_size = index_sz;
    g_nchan = num_chans;
    g_num_subbands = num_subbands;

    /* since CUDASafeCall() calls cudaGetErrorString(),
       it should not be used here - will cause crash if no CUDA device is
       found */
    (void) cudaGetDeviceCount(&iDevCount);
    if (0 == iDevCount)
    {
        (void) fprintf(stderr, "ERROR: No CUDA-capable device found!\n");
        run = 0;
        return EXIT_FAILURE;
    }

    /* just use the first device */
    CUDASafeCall(cudaSetDevice(0));

    CUDASafeCall(cudaGetDeviceProperties(&stDevProp, 0));
    g_iMaxThreadsPerBlock = stDevProp.maxThreadsPerBlock;

    g_pcPFBCoeff = (signed char *) malloc(g_num_subbands
                                          * VEGAS_NUM_TAPS
                                          * g_nchan
                                          * sizeof(signed char));
    if (NULL == g_pcPFBCoeff)
    {
        (void) fprintf(stderr,
                       "ERROR: Memory allocation failed! %s.\n",
                       strerror(errno));
        return EXIT_FAILURE;
    }

    /* allocate memory for the filter coefficient array on the device */
    CUDASafeCall(cudaMalloc((void **) &g_pcPFBCoeff_d,
                                       g_num_subbands
                                       * VEGAS_NUM_TAPS
                                       * g_nchan
                                       * sizeof(signed char)));

    /* read filter coefficients */
    /* build file name */
    (void) sprintf(g_acFileCoeff,
                   "%s_%s_%d_%d_%d%s",
                   FILE_COEFF_PREFIX,
                   FILE_COEFF_DATATYPE,
                   VEGAS_NUM_TAPS,
                   g_nchan,
                   g_num_subbands,
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
                g_num_subbands * VEGAS_NUM_TAPS * g_nchan * sizeof(signed char));
    if (iRet != (g_num_subbands * VEGAS_NUM_TAPS * g_nchan * sizeof(signed char)))
    {
        (void) fprintf(stderr,
                       "ERROR: Reading filter coefficients failed! %s.\n",
                       strerror(errno));
        return EXIT_FAILURE;
    }
    (void) close(g_iFileCoeff);

    /* copy filter coefficients to the device */
    CUDASafeCall(cudaMemcpy(g_pcPFBCoeff_d,
               g_pcPFBCoeff,
               g_num_subbands * VEGAS_NUM_TAPS * g_nchan * sizeof(signed char),
               cudaMemcpyHostToDevice));

    /* allocate memory for data array - 32MB is the block size for the VEGAS
       input buffer, allocate 32MB + space for (VEGAS_NUM_TAPS - 1) blocks of
       data
       NOTE: the actual data in a 32MB block will be only
       (num_heaps * heap_size), but since we don't know that value until data
       starts flowing, allocate the maximum possible size */
    CUDASafeCall(cudaMalloc((void **) &g_pc4Data_d,
                                       (g_buf_in_block_size
                                        + ((VEGAS_NUM_TAPS - 1)
                                           * g_num_subbands
                                           * g_nchan
                                           * sizeof(char4)))));
    g_pc4DataRead_d = g_pc4Data_d;

    /* calculate kernel parameters */
    /* ASSUMPTION: g_nchan >= g_iMaxThreadsPerBlock */
    g_dimBPFB.x = g_iMaxThreadsPerBlock;
    g_dimBAccum.x = g_iMaxThreadsPerBlock;
    g_dimGPFB.x = (g_num_subbands * g_nchan) / g_iMaxThreadsPerBlock;
    g_dimGAccum.x = (g_num_subbands * g_nchan) / g_iMaxThreadsPerBlock;

    CUDASafeCall(cudaMalloc((void **) &g_pf4FFTIn_d,
                                 g_num_subbands * g_nchan * sizeof(float4)));
    CUDASafeCall(cudaMalloc((void **) &g_pf4FFTOut_d,
                                 g_num_subbands * g_nchan * sizeof(float4)));
    CUDASafeCall(cudaMalloc((void **) &g_pf4SumStokes_d,
                                 g_num_subbands * g_nchan * sizeof(float4)));
    CUDASafeCall(cudaMemset(g_pf4SumStokes_d,
                                 '\0',
                                 g_num_subbands * g_nchan * sizeof(float4)));

    /* create plan */
    iCUFFTRet = cufftPlanMany(&g_stPlan,
                              FFTPLAN_RANK,
                              &g_nchan,
                              &g_nchan,
                              FFTPLAN_ISTRIDE,
                              FFTPLAN_IDIST,
                              &g_nchan,
                              FFTPLAN_OSTRIDE,
                              FFTPLAN_ODIST,
                              CUFFT_C2C,
                              FFTPLAN_BATCH);
    if (iCUFFTRet != CUFFT_SUCCESS)
    {
        (void) fprintf(stderr, "ERROR: Plan creation failed!\n");
        run = 0;
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}

/* Actually do the PFB by calling CUDA kernels */
extern "C"
void do_pfb(struct guppi_databuf *db_in,
            int curblock_in,
            struct guppi_databuf *db_out,
            int first,
            struct guppi_status st,
            int acc_len)
{
    /* Declare local variables */
    char *hdr_out = NULL;
    struct databuf_index *index_in = NULL;
    struct databuf_index *index_out = NULL;
    int heap_in = 0;
    char *heap_addr_in = NULL;
    char *heap_addr_out = NULL;
    struct time_spead_heap* first_time_heap_in_accum = NULL;
    struct freq_spead_heap* freq_heap_out = NULL;
    int iProcData = 0;
    cudaError_t iCUDARet = cudaSuccess;
    int iRet = GUPPI_OK;
    char* payload_addr_in = NULL;
    char* payload_addr_out = NULL;
    int num_in_heaps_per_proc = 0;
    int num_in_heaps_per_acc = 0;
    int pfb_count = 0;
    int num_in_heaps_gpu_buffer = 0;
    int num_in_heaps_tail = 0;
    int i = 0;

    /* Setup input and first output data block stuff */
    index_in = (struct databuf_index*)guppi_databuf_index(db_in, curblock_in);
    /* Get the number of heaps per block of data that will be processed by the GPU */
    num_in_heaps_per_proc = (g_num_subbands * g_nchan * sizeof(char4)) / (index_in->heap_size - sizeof(struct time_spead_heap));
    num_in_heaps_per_acc = num_in_heaps_per_proc * acc_len;
    g_block_in_data_size = (index_in->num_heaps * index_in->heap_size) - (index_in->num_heaps * sizeof(struct time_spead_heap));

    num_in_heaps_tail = (((VEGAS_NUM_TAPS - 1) * g_num_subbands * g_nchan * sizeof(char4))
                         / (index_in->heap_size - sizeof(struct time_spead_heap)));
    num_in_heaps_gpu_buffer = index_in->num_heaps + num_in_heaps_tail;

    /* Calculate the maximum number of output heaps per block */
    g_iMaxNumHeapOut = (g_buf_out_block_size - (sizeof(struct time_spead_heap) * MAX_HEAPS_PER_BLK)) / (g_num_subbands * g_nchan * sizeof(float4)); 

    hdr_out = guppi_databuf_header(db_out, g_pfb_curblock_out);
    index_out = (struct databuf_index*)guppi_databuf_index(db_out, g_pfb_curblock_out);
    memcpy(hdr_out, guppi_databuf_header(db_in, curblock_in),
            GUPPI_STATUS_SIZE);

    /* Set basic params in output index */
    index_out->heap_size = sizeof(struct freq_spead_heap) + (g_num_subbands * g_nchan * sizeof(float4));
    /* Read in heap from buffer */
    heap_addr_in = (char*)(guppi_databuf_data(db_in, curblock_in) +
                        sizeof(struct time_spead_heap) * heap_in);
    first_time_heap_in_accum = (struct time_spead_heap*)(heap_addr_in);
    if (first)
    {
        g_first_heap_in = heap_in;
        printf("***** pfb_gpu: g_first_heap_in = %d\n", g_first_heap_in);
        g_first_heap_rcvd_mjd = index_in->cpu_gpu_buf[g_first_heap_in].heap_rcvd_mjd;
    }
    /* Here, the payload_addr_in is the start of the contiguous block of data that will be
       copied to the GPU (heap_in = 0) */
    payload_addr_in = (char*)(guppi_databuf_data(db_in, curblock_in) +
                        sizeof(struct time_spead_heap) * MAX_HEAPS_PER_BLK +
                        (index_in->heap_size - sizeof(struct time_spead_heap)) * heap_in );

    /* Copy data block to GPU */
    if (first)
    {
        /* Sanity check for the first iteration */
        if ((g_block_in_data_size % (g_num_subbands * g_nchan * sizeof(char4))) != 0)
        {
            (void) fprintf(stderr, "ERROR: Data size mismatch!\n");
            run = 0;
            return;
        }
        CUDASafeCall(cudaMemcpy(g_pc4Data_d,
                                payload_addr_in,
                                g_block_in_data_size,
                                cudaMemcpyHostToDevice));
        /* duplicate the last (VEGAS_NUM_TAPS - 1) segments at the end for
           the next iteration */
        CUDASafeCall(cudaMemcpy(g_pc4Data_d + (g_block_in_data_size / sizeof(char4)),
                                g_pc4Data_d + (g_block_in_data_size / sizeof(char4)) - ((VEGAS_NUM_TAPS - 1) * g_num_subbands * g_nchan),
                                ((VEGAS_NUM_TAPS - 1) * g_num_subbands * g_nchan * sizeof(char4)),
                                cudaMemcpyDeviceToDevice));

        /* copy the status bits and valid flags for all heaps to arrays separate
           from the index, so that it can be combined with the corresponding
           values from the previous block */
        struct time_spead_heap* time_heap = (struct time_spead_heap*) guppi_databuf_data(db_in, curblock_in);
        for (i = 0; i < index_in->num_heaps; ++i)
        {
            g_auiStatusBits[i] = time_heap->status_bits;
            g_auiHeapValid[i] = index_in->cpu_gpu_buf[i].heap_valid;
            ++time_heap;
        }
        /* duplicate the last (VEGAS_NUM_TAPS - 1) segments at the end for the
           next iteration */
        for ( ; i < index_in->num_heaps + num_in_heaps_tail; ++i)
        {
            g_auiStatusBits[i] = g_auiStatusBits[i-num_in_heaps_tail];
            g_auiHeapValid[i] = g_auiHeapValid[i-num_in_heaps_tail];
        }
    }
    else
    {
        /* If this is not the first run, need to handle block boundary, while doing the PFB */
        CUDASafeCall(cudaMemcpy(g_pc4Data_d,
                                g_pc4Data_d + (g_block_in_data_size / sizeof(char4)),
                                ((VEGAS_NUM_TAPS - 1) * g_num_subbands * g_nchan * sizeof(char4)),
                                cudaMemcpyDeviceToDevice));
        CUDASafeCall(cudaMemcpy(g_pc4Data_d + ((VEGAS_NUM_TAPS - 1) * g_num_subbands * g_nchan),
                                payload_addr_in,
                                g_block_in_data_size,
                                cudaMemcpyHostToDevice));
        /* copy the status bits and valid flags for all heaps to arrays separate
           from the index, so that it can be combined with the corresponding
           values from the previous block */
        for (i = 0; i < num_in_heaps_tail; ++i)
        {
            g_auiStatusBits[i] = g_auiStatusBits[index_in->num_heaps+i];
            g_auiHeapValid[i] = g_auiHeapValid[index_in->num_heaps+i];
        }
        struct time_spead_heap* time_heap = (struct time_spead_heap*) guppi_databuf_data(db_in, curblock_in);
        for ( ; i < num_in_heaps_tail + index_in->num_heaps; ++i)
        {
            g_auiStatusBits[i] = time_heap->status_bits;
            g_auiHeapValid[i] = index_in->cpu_gpu_buf[i-num_in_heaps_tail].heap_valid;
            ++time_heap;
        }
    }

    g_pc4DataRead_d = g_pc4Data_d;
    iProcData = 0;
    while (g_block_in_data_size != iProcData)  /* loop till (num_heaps * heap_size) of data is processed */
    {
        if (0 == pfb_count)
        {
            /* Check if all heaps necessary for this PFB are valid */
            if (!(is_valid(heap_in, (VEGAS_NUM_TAPS * num_in_heaps_per_proc))))
            {
                /* Skip all heaps that go into this PFB if there is an invalid heap */
                iProcData += (VEGAS_NUM_TAPS * g_num_subbands * g_nchan * sizeof(char4));
                /* update the data read pointer */
                g_pc4DataRead_d += (VEGAS_NUM_TAPS * g_num_subbands * g_nchan);
                if (iProcData == g_block_in_data_size)
                {
                    break;
                }

                /* Calculate input heap addresses for the next round of processing */
                heap_in += (VEGAS_NUM_TAPS * num_in_heaps_per_proc);
                if (heap_in > num_in_heaps_gpu_buffer)
                {
                    /* This is not supposed to happen */
                    (void) fprintf(stderr,
                                   "ERROR: Heap count %d exceeds available number of heaps %d!\n",
                                   heap_in,
                                   num_in_heaps_gpu_buffer);
                    run = 0;
                    return;
                }
                heap_addr_in = (char*)(guppi_databuf_data(db_in, curblock_in) +
                                    sizeof(struct time_spead_heap) * heap_in);
                continue;
            }
        }

        /* Perform polyphase filtering */
        DoPFB<<<g_dimGPFB, g_dimBPFB>>>(g_pc4DataRead_d,
                                        g_pf4FFTIn_d,
                                        g_pcPFBCoeff_d);
        CUDASafeCall(cudaThreadSynchronize());
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

        /* Accumulate power x, power y, stokes real and imag, if the blanking
           bit is not set */
        if (!(is_blanked(heap_in, num_in_heaps_per_proc)))
        {
            iRet = accumulate();
            if (iRet != GUPPI_OK)
            {
                (void) fprintf(stderr, "ERROR: Accumulation failed!\n");
                run = 0;
                break;
            }
            ++g_iSpecPerAcc;
            g_iPrevBlankingState = FALSE;
        }
        else
        {
            /* state just changed */
            if (FALSE == g_iPrevBlankingState)
            {
                /* dump to buffer */
                heap_addr_out = (char*)(guppi_databuf_data(db_out, g_pfb_curblock_out) +
                                    sizeof(struct freq_spead_heap) * g_heap_out);
                freq_heap_out = (struct freq_spead_heap*)(heap_addr_out);
                payload_addr_out = (char*)(guppi_databuf_data(db_out, g_pfb_curblock_out) +
                                    sizeof(struct freq_spead_heap) * MAX_HEAPS_PER_BLK +
                                    (index_out->heap_size - sizeof(struct freq_spead_heap)) * g_heap_out);
         
                /* Write new heap header fields */
                freq_heap_out->time_cntr_id = 0x20;
                freq_heap_out->time_cntr = first_time_heap_in_accum->time_cntr;
                freq_heap_out->spectrum_cntr_id = 0x21;
                freq_heap_out->spectrum_cntr = g_tot_heap_out;
                freq_heap_out->integ_size_id = 0x22;
                freq_heap_out->integ_size = g_iSpecPerAcc;
                freq_heap_out->mode_id = 0x23;
                freq_heap_out->mode = first_time_heap_in_accum->mode;
                freq_heap_out->status_bits_id = 0x24;
                freq_heap_out->status_bits = first_time_heap_in_accum->status_bits;
                freq_heap_out->payload_data_off_addr_mode = 0;
                freq_heap_out->payload_data_off_id = 0x25;
                freq_heap_out->payload_data_off = 0;

                /* Update output index */
                index_out->cpu_gpu_buf[g_heap_out].heap_valid = 1;
                index_out->cpu_gpu_buf[g_heap_out].heap_cntr = g_tot_heap_out;
                index_out->cpu_gpu_buf[g_heap_out].heap_rcvd_mjd =
                         index_in->cpu_gpu_buf[g_first_heap_in].heap_rcvd_mjd ;

                iRet = get_accumulated_spectrum_from_device(payload_addr_out);
                if (iRet != GUPPI_OK)
                {
                    (void) fprintf(stderr, "ERROR: Getting accumulated spectrum failed!\n");
                    run = 0;
                    break;
                }

                ++g_heap_out;
                ++g_tot_heap_out;

                /* zero accumulators */
                zero_accumulator();
                /* reset time */
                g_iSpecPerAcc = 0;
                g_iPrevBlankingState = TRUE;
            }
        }

        if (g_iSpecPerAcc == acc_len)
        {
            /* dump to buffer */
            heap_addr_out = (char*)(guppi_databuf_data(db_out, g_pfb_curblock_out) +
                                sizeof(struct freq_spead_heap) * g_heap_out);
            freq_heap_out = (struct freq_spead_heap*)(heap_addr_out);
            payload_addr_out = (char*)(guppi_databuf_data(db_out, g_pfb_curblock_out) +
                                sizeof(struct freq_spead_heap) * MAX_HEAPS_PER_BLK +
                                (index_out->heap_size - sizeof(struct freq_spead_heap)) * g_heap_out);
     
            /* Write new heap header fields */
            freq_heap_out->time_cntr_id = 0x20;
            freq_heap_out->time_cntr = first_time_heap_in_accum->time_cntr;
            freq_heap_out->spectrum_cntr_id = 0x21;
            freq_heap_out->spectrum_cntr = g_tot_heap_out;
            freq_heap_out->integ_size_id = 0x22;
            freq_heap_out->integ_size = g_iSpecPerAcc;
            freq_heap_out->mode_id = 0x23;
            freq_heap_out->mode = first_time_heap_in_accum->mode;
            freq_heap_out->status_bits_id = 0x24;
            freq_heap_out->status_bits = first_time_heap_in_accum->status_bits;
            freq_heap_out->payload_data_off_addr_mode = 0;
            freq_heap_out->payload_data_off_id = 0x25;
            freq_heap_out->payload_data_off = 0;

            /* Update output index */
            index_out->cpu_gpu_buf[g_heap_out].heap_valid = 1;
            index_out->cpu_gpu_buf[g_heap_out].heap_cntr = g_tot_heap_out;
            index_out->cpu_gpu_buf[g_heap_out].heap_rcvd_mjd =
                     index_in->cpu_gpu_buf[g_first_heap_in].heap_rcvd_mjd ;

            iRet = get_accumulated_spectrum_from_device(payload_addr_out);
            if (iRet != GUPPI_OK)
            {
                (void) fprintf(stderr, "ERROR: Getting accumulated spectrum failed!\n");
                run = 0;
                break;
            }

            ++g_heap_out;
            ++g_tot_heap_out;

            /* zero accumulators */
            zero_accumulator();
            /* reset time */
            g_iSpecPerAcc = 0;
        }

        iProcData += (g_num_subbands * g_nchan * sizeof(char4));
        /* update the data read pointer */
        g_pc4DataRead_d += (g_num_subbands * g_nchan);

        /* Calculate input heap addresses for the next round of processing */
        heap_in += num_in_heaps_per_proc;
        heap_addr_in = (char*)(guppi_databuf_data(db_in, curblock_in) +
                            sizeof(struct time_spead_heap) * heap_in);
        if (0 == g_iSpecPerAcc)
        {
            first_time_heap_in_accum = (struct time_spead_heap*)(heap_addr_in);
            g_first_heap_in = heap_in;
            g_first_heap_rcvd_mjd = index_in->cpu_gpu_buf[g_first_heap_in].heap_rcvd_mjd;
        }

        /* if output block is full */
        if (g_heap_out == g_iMaxNumHeapOut)
        {
            /* Set the number of heaps written to this block */
            index_out->num_heaps = g_heap_out;

            /* Mark output buffer as filled */
            guppi_databuf_set_filled(db_out, g_pfb_curblock_out);

            printf("Debug: vegas_pfb_thread going to next output block\n");

            /* Note current output block */
            //guppi_status_lock_safe(&st);
                pthread_cleanup_push((void (*) (void *))&guppi_status_unlock, (void *) &st);
                guppi_status_lock(&st);
            hputi4(st.buf, "PFBBLKOU", g_pfb_curblock_out);
            //guppi_status_unlock_safe(&st);
               guppi_status_unlock(&st);
               pthread_cleanup_pop(0);

            /*  Wait for next output block */
            g_pfb_curblock_out = (g_pfb_curblock_out + 1) % db_out->n_block;
            while ((guppi_databuf_wait_free(db_out, g_pfb_curblock_out)!=0) && run) {
                //guppi_status_lock_safe(&st);
                    pthread_cleanup_push((void (*)(void *))&guppi_status_unlock, (void *) &st);
                    guppi_status_lock(&st);

                hputs(st.buf, STATUS_KEY, "blocked");
                //guppi_status_unlock_safe(&st);
                   guppi_status_unlock(&st);
                   pthread_cleanup_pop(0);
            }

            g_heap_out = 0;

            hdr_out = guppi_databuf_header(db_out, g_pfb_curblock_out);
            index_out = (struct databuf_index*)guppi_databuf_index(db_out, g_pfb_curblock_out);
            memcpy(hdr_out, guppi_databuf_header(db_in, curblock_in),
                    GUPPI_STATUS_SIZE);

            /* Set basic params in output index */
            index_out->heap_size = sizeof(struct freq_spead_heap) + (g_num_subbands * g_nchan * sizeof(float4));
        }

        pfb_count = (pfb_count + 1) % VEGAS_NUM_TAPS;
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
                                             g_pf4SumStokes_d);
    CUDASafeCall(cudaThreadSynchronize());
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
    CUDASafeCall(cudaMemset(g_pf4SumStokes_d,
                                       '\0',
                                       (g_num_subbands
                                       * g_nchan
                                       * sizeof(float4))));

    return;
}

int get_accumulated_spectrum_from_device(char *out)
{
    CUDASafeCall(cudaMemcpy(out,
                                       g_pf4SumStokes_d,
                                       (g_num_subbands
                                        * g_nchan
                                        * sizeof(float4)),
                                       cudaMemcpyDeviceToHost));

    return GUPPI_OK;
}

/*
 * function to be used to check if any heap within the current PFB is invalid,
 * in which case, the entire PFB should be discarded.
 * NOTE: this function does not check ALL heaps - it returns at the first
 * invalid heap.
 */
int is_valid(int heap_start, int num_heaps)
{
    for (int i = heap_start; i < (heap_start + num_heaps); ++i)
    {
        if (!g_auiHeapValid[i])
        {
            return FALSE;
        }
    }

    return TRUE;
}

/*
 * function that checks if blanking has started within this accumulation.
 * ASSUMPTION: the blanking bit does not toggle within this time interval.
 */
int is_blanked(int heap_start, int num_heaps)
{
    for (int i = heap_start; i < (heap_start + num_heaps); ++i)
    {
        if (g_auiStatusBits[i] & 0x08)
        {
            return TRUE;
        }
    }

    return FALSE;
}

void __CUDASafeCall(cudaError_t iCUDARet,
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

    /* destroy plan */
    /* TODO: check if plan exists */
    (void) cufftDestroy(g_stPlan);

    return;
}

