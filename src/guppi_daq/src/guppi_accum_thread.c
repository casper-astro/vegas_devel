/* guppi_accum_thread.c
 *
 * Adds the heaps, received from the network, to the appropriate accumulators.
 * At set intervals (e.g. every 1 second) the accumulators are dumped to the output
 * buffer, for writing to the disk.
 */

#define _GNU_SOURCE 1
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <pthread.h>
#include <signal.h>
#include <sched.h>
#include <sys/time.h>
#include <sys/resource.h>
#include <sys/types.h>

#include "guppi_defines.h"
#include "fitshead.h"
#include "sdfits.h"
#include "guppi_error.h"
#include "guppi_status.h"
#include "guppi_databuf.h"
#include "spead_heap.h"

#define STATUS_KEY "ACCSTAT"
#include "guppi_threads.h"

#define NUM_SW_STATES   8
#define MAX_NUM_SUB     8
#define MAX_NUM_CH      32768
#define NUM_STOKES      4

// Read a status buffer all of the key observation paramters
extern void guppi_read_obs_params(char *buf, 
                                  struct guppi_params *g, 
                                  struct sdfits *p);

/* Parse info from buffer into param struct */
extern void guppi_read_subint_params(char *buf, 
                                     struct guppi_params *g,
                                     struct sdfits *p);

/* Read the PFB rate */
void guppi_read_pfbrate(char *buf, float *pfb_rate);


/* Resets the vector accumulators */
void reset_accumulators(float accumulator[][MAX_NUM_SUB][MAX_NUM_CH][NUM_STOKES],
                        struct sdfits_data_columns* data_cols, char* accum_dirty,
                        int num_subbands, int num_chans)
{
    int i, j, k, l;
    int sum = 0;

    for(i = 0; i < NUM_SW_STATES; i++)
    {
        if(accum_dirty[i])
        {
            for(j = 0; j < num_subbands; j++)
            {
                for(k = 0; k < num_chans; k++)
                {
                    for(l = 0; l < NUM_STOKES; l++)
                        accumulator[i][j][k][l] = 0.0;
                }
            }

            data_cols[i].time = 0.0;
            data_cols[i].exposure = 0.0;
            data_cols[i].sttspec = sum;
            data_cols[i].stpspec = 0;
            data_cols[i].data = NULL;

            accum_dirty[i] = 0;
        }

    } 
}


/* Declare accumulators */
float accumulator[NUM_SW_STATES][MAX_NUM_SUB][MAX_NUM_CH][NUM_STOKES];


/* The main CPU accumulator thread */
void guppi_accum_thread(void *_args) {

    char accum_dirty[NUM_SW_STATES];
    struct sdfits_data_columns data_cols[NUM_SW_STATES];

    /* Get arguments */
    struct guppi_thread_args *args = (struct guppi_thread_args *)_args;

    /* Set cpu affinity */
    cpu_set_t cpuset;
    sched_getaffinity(0, sizeof(cpu_set_t), &cpuset);
    CPU_CLR(2, &cpuset);
    CPU_CLR(3, &cpuset);
    int rv = sched_setaffinity(0, sizeof(cpu_set_t), &cpuset);
    if (rv<0) { 
        guppi_error("guppi_accum_thread", "Error setting cpu affinity.");
        perror("sched_setaffinity");
    }

    /* Set priority */
    rv = setpriority(PRIO_PROCESS, 0, args->priority);
    if (rv<0) {
        guppi_error("guppi_accum_thread", "Error setting priority level.");
        perror("set_priority");
    }

    /* Attach to status shared mem area */
    struct guppi_status st;
    rv = guppi_status_attach(&st);
    if (rv!=GUPPI_OK) {
        guppi_error("guppi_accum_thread", 
                "Error attaching to status shared memory.");
        pthread_exit(NULL);
    }
    pthread_cleanup_push((void *)guppi_status_detach, &st);
    pthread_cleanup_push((void *)set_exit_status, &st);
    pthread_cleanup_push((void *)guppi_thread_set_finished, args);

    /* Init status */
    guppi_status_lock_safe(&st);
    hputs(st.buf, STATUS_KEY, "init");
    guppi_status_unlock_safe(&st);

    /* Read in general parameters */
    struct guppi_params gp;
    struct sdfits sf;
    pthread_cleanup_push((void *)guppi_free_sdfits, &sf);

    /* Attach to databuf shared mem */
    struct guppi_databuf *db_in, *db_out;
    db_in = guppi_databuf_attach(args->input_buffer);
    char errmsg[256];
    if (db_in==NULL) {
        sprintf(errmsg,
                "Error attaching to input databuf(%d) shared memory.", 
                args->input_buffer);
        guppi_error("guppi_accum_thread", errmsg);
        pthread_exit(NULL);
    }
    pthread_cleanup_push((void *)guppi_databuf_detach, db_in);
    db_out = guppi_databuf_attach(args->output_buffer);
    if (db_out==NULL) {
        sprintf(errmsg,
                "Error attaching to output databuf(%d) shared memory.", 
                args->output_buffer);
        guppi_error("guppi_accum_thread", errmsg);
        pthread_exit(NULL);
    }
    pthread_cleanup_push((void *)guppi_databuf_detach, db_out);

    /* Loop */
    int curblock_in=0, curblock_out=0;
    int first=1;
    float reqd_exposure=0;
    unsigned int accum_stop_time=0;
    float pfb_rate;
    int heap, accumid, i, j,k, struct_offset, array_offset;
    char *hdr_in=NULL, *hdr_out=NULL;
    struct databuf_index *index_in, *index_out;

    int nblock_int=0, npacket=0, n_pkt_drop=0, n_heap_drop=0;

    signal(SIGINT,cc);
    while (run) {

        /* Note waiting status */
        guppi_status_lock_safe(&st);
        hputs(st.buf, STATUS_KEY, "waiting");
        guppi_status_unlock_safe(&st);

        /* Wait for buf to have data */
        rv = guppi_databuf_wait_filled(db_in, curblock_in);
        if (rv!=0) continue;

        /* Note current block(s) */
        guppi_status_lock_safe(&st);
        hputi4(st.buf, "ACCBLKIN", curblock_in);
        guppi_status_unlock_safe(&st);

        /* Note waiting status */
        guppi_status_lock_safe(&st);
        hputs(st.buf, STATUS_KEY, "accumulating");
        guppi_status_unlock_safe(&st);

        /* Read param struct for this block */
        hdr_in = guppi_databuf_header(db_in, curblock_in);
        if (first) 
            guppi_read_obs_params(hdr_in, &gp, &sf);
        else
            guppi_read_subint_params(hdr_in, &gp, &sf);

        /* Do any first time stuff: first time code runs, not first time process this block*/
        if (first) {

            /* Set up first output header. This header is copied from block to block
               each time a new block is created */
            hdr_out = guppi_databuf_header(db_out, curblock_out);
            memcpy(hdr_out, guppi_databuf_header(db_in, curblock_in),
                    GUPPI_STATUS_SIZE);

            /* Read required exposure and PFB rate from status shared memory */
            reqd_exposure = sf.data_columns.exposure;
            guppi_read_pfbrate(st.buf, &pfb_rate);

            accum_stop_time = (unsigned int)(reqd_exposure/1e-3);

            /* Initialise the index in the output block */
            index_out = (struct databuf_index*)guppi_databuf_index(db_out, curblock_out);
            index_out->num_datasets = 0;
            index_out->array_size = sf.hdr.nsubband * sf.hdr.nchan * NUM_STOKES * 4;

            /* Clear the vector accumulators */
            for(i = 0; i < NUM_SW_STATES; i++) accum_dirty[i] = 1;
            reset_accumulators(accumulator, data_cols, accum_dirty,
                                sf.hdr.nsubband, sf.hdr.nchan);

            first=0;
        }

        /* Loop through each spectrum (heap) in input buffer */
        index_in = (struct databuf_index*)guppi_databuf_index(db_in, curblock_in);

        for(heap = 0; heap < index_in->num_heaps; heap++)
        {
            /* If invalid, record it and move to next heap */
            if(!index_in->cpu_gpu_buf[heap].heap_valid)
            {
                n_heap_drop++;
                continue;
            }

            /* Read in heap from buffer */
            char* heap_addr = (char*)(guppi_databuf_data(db_in, curblock_in) +
                                (index_in->heap_size)*heap);
            struct freq_spead_heap* freq_heap = (struct freq_spead_heap*)(heap_addr);

            int *payload = (int*)(heap_addr + sizeof(struct freq_spead_heap));

            accumid = freq_heap->status_bits & 0x7;         

            /*Debug: print heap */
/*            printf("%d, %d, %d, %d, %d, %d\n", freq_heap->time_cntr, freq_heap->spectrum_cntr,
                freq_heap->integ_size, freq_heap->mode, freq_heap->status_bits,
                freq_heap->payload_data_off);
*/

            /* If we have accumulated for long enough, write vectors to output block */
            if(freq_heap->spectrum_cntr >= accum_stop_time)
            {
                for(i = 0; i < NUM_SW_STATES; i++)
                {
                    /*If a particular accumulator is dirty, write it to output buffer */
                    if(accum_dirty[i])
                    {
                        /*If insufficient space, first mark block as filled and request new block*/
                        index_out = (struct databuf_index*)(guppi_databuf_index(db_out, curblock_out));

                        if( (index_out->num_datasets+1) *
                            (index_out->array_size + sizeof(struct sdfits_data_columns)) > 
                            db_out->block_size)
                        {
                            printf("Accumulator finished with output block %d\n", curblock_out);

                            /* Write block number to status buffer */
                            guppi_status_lock_safe(&st);
                            hputi4(st.buf, "ACCBLKOU", curblock_out);
                            guppi_status_unlock_safe(&st);

                            /* Update packet count and loss fields in output header */
                            hputi4(hdr_out, "NBLOCK", nblock_int);
                            hputi4(hdr_out, "NPKT", npacket);
                            hputi4(hdr_out, "NPKTDROP", n_pkt_drop);
                            hputi4(hdr_out, "NHPDROP", n_heap_drop);

                            /* Close out current integration */
                            guppi_databuf_set_filled(db_out, curblock_out);

                            /* Wait for next output buf */
                            curblock_out = (curblock_out + 1) % db_out->n_block;
                            guppi_databuf_wait_free(db_out, curblock_out);
                            hdr_out = guppi_databuf_header(db_out, curblock_out);
                            memcpy(hdr_out, guppi_databuf_header(db_in, curblock_in),
                                    GUPPI_STATUS_SIZE);

                            /* Initialise the index in new output block */
                            index_out = (struct databuf_index*)guppi_databuf_index(db_out, curblock_out);
                            index_out->num_datasets = 0;
                            index_out->array_size = sf.hdr.nsubband * sf.hdr.nchan * NUM_STOKES * 4;
                            
                            nblock_int=0;
                            npacket=0;
                            n_pkt_drop=0;
                            n_heap_drop=0;
                        }            

                        /*Update index for output buffer*/
                        index_out = (struct databuf_index*)(guppi_databuf_index(db_out, curblock_out));

                        if(index_out->num_datasets == 0)
                            struct_offset = 0;
                        else
                            struct_offset = index_out->disk_buf[index_out->num_datasets-1].array_offset +
                                            index_out->array_size;

                        array_offset =  struct_offset + sizeof(struct sdfits_data_columns);
                        index_out->disk_buf[index_out->num_datasets].struct_offset = struct_offset;
                        index_out->disk_buf[index_out->num_datasets].array_offset = array_offset;

                        /*Copy sdfits_data_columns struct to disk buffer */
                        memcpy(guppi_databuf_data(db_out, curblock_out) + struct_offset,
                                &data_cols[i], sizeof(struct sdfits_data_columns));

                        /*Copy data array to disk buffer */
                        memcpy(guppi_databuf_data(db_out, curblock_out) + array_offset,
                                accumulator[accumid], index_out->array_size);
                        
                        /*Update SDFITS data_columns pointer to data array */
                        ((struct sdfits_data_columns*)
                        (guppi_databuf_data(db_out, curblock_out) + struct_offset))->data = 
                        (unsigned char*)(guppi_databuf_data(db_out, curblock_out) + array_offset);

                        index_out->num_datasets = index_out->num_datasets + 1;

                    }
                
                }

                accum_stop_time += (unsigned int)(reqd_exposure/1e-3);

                reset_accumulators(accumulator, data_cols, accum_dirty,
                                sf.hdr.nsubband, sf.hdr.nchan);
            }

            /* Only add spectrum to accumulator if blanking bit is low */
            if((freq_heap->status_bits & 0x08) == 0)
            {
                /* Fill in data columns header fields */
                if(!accum_dirty[accumid])
                {
                    /*Record SPEAD header fields*/
                    data_cols[accumid].time = (double)(freq_heap->time_cntr);
                    data_cols[accumid].sttspec = freq_heap->spectrum_cntr;
                    data_cols[accumid].accumid = accumid;

                    /* Fill in rest of fields from status buffer */
                    strcpy(data_cols[accumid].object, sf.data_columns.object);
                    data_cols[accumid].azimuth = sf.data_columns.azimuth;
                    data_cols[accumid].elevation = sf.data_columns.elevation;
                    data_cols[accumid].bmaj = sf.data_columns.bmaj;
                    data_cols[accumid].bmin = sf.data_columns.bmin;
                    data_cols[accumid].bpa = sf.data_columns.bpa;
                    data_cols[accumid].centre_freq_idx = sf.data_columns.centre_freq_idx;
                    data_cols[accumid].ra = sf.data_columns.ra;
                    data_cols[accumid].dec = sf.data_columns.dec;

                    for(i = 0; i < NUM_SW_STATES; i++)
                        data_cols[accumid].centre_freq[i] = sf.data_columns.centre_freq[i];

                    accum_dirty[accumid] = 1;
                }

                data_cols[accumid].exposure += (float)(freq_heap->integ_size)/pfb_rate;
                data_cols[accumid].stpspec = freq_heap->spectrum_cntr;

                /* Add spectrum to appropriate vector accumulator */
                for(i = 0; i < sf.hdr.nsubband; i++)
                {
                    for(j = 0; j < sf.hdr.nchan; j++)
                    {
                        for(k = 0; k < NUM_STOKES; k++)
                        {
                            accumulator[accumid][i][j][k] +=
                                (float)payload[i*sf.hdr.nchan + j*NUM_STOKES + k];
                        }
                    }
                }
            }
            
        }

        /* Update packet count and loss fields from input header */
        nblock_int++;
        npacket += gp.num_pkts_rcvd;
        n_pkt_drop += gp.num_pkts_dropped;

        /* Done with current input block */
        guppi_databuf_set_free(db_in, curblock_in);
        curblock_in = (curblock_in + 1) % db_in->n_block;

        /* Check for cancel */
        pthread_testcancel();
    }

    pthread_exit(NULL);
    pthread_cleanup_pop(0); /* Closes set_exit_status */
    pthread_cleanup_pop(0); /* Closes set_finished */
    pthread_cleanup_pop(0); /* Closes guppi_free_sdfits */
    pthread_cleanup_pop(0); /* Closes ? */
    pthread_cleanup_pop(0); /* Closes guppi_status_detach */
    pthread_cleanup_pop(0); /* Closes guppi_databuf_detach */
}
