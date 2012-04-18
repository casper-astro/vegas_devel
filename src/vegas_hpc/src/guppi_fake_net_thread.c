/* guppi_fake_net_thread.c
 *
 * Routine to write fake data into shared memory blocks.
 * This allows the processing pipelines and PSRFITS routines to be
 * tested without the network portion of GUPPI.
 */

#define _GNU_SOURCE 1
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <unistd.h>
#include <string.h>
#include <pthread.h>
#include <signal.h>
#include <sched.h>
#include <sys/time.h>
#include <sys/resource.h>
#include <sys/types.h>

#include "fitshead.h"
#include "guppi_params.h"
#include "guppi_error.h"
#include "guppi_status.h"
#include "guppi_databuf.h"
#include "guppi_udp.h"
#include "guppi_time.h"
#include "spead_heap.h"

#define STATUS_KEY "NETSTAT"  /* Define before guppi_threads.h */
#include "guppi_threads.h"
#include "guppi_defines.h"

#if FITS_TYPE == PSRFITS
#include "psrfits.h"
#else
#include "sdfits.h"
#endif

#define MAX_HEAPS_IN_BLOCK  2048
#define SPEAD_SPECTRUM_SZ   1024

// Read a status buffer all of the key observation paramters
#if FITS_TYPE == PSRFITS
extern void guppi_read_obs_params(char *buf, 
                                  struct guppi_params *g, 
                                  struct psrfits *p);
#else
extern void guppi_read_obs_params(char *buf, 
                                  struct guppi_params *g, 
                                  struct sdfits *p);
#endif

/* Structs/functions to more easily deal with multiple 
 * active blocks being filled
 */
struct fake_datablock_stats {
    struct guppi_databuf *db;       // Pointer to overall shared mem databuf
    int block_idx;                  // Block index number in databuf
    unsigned long long heap_idx;    // Index of first heap number in block
    size_t heap_size;               // Data size of each heap
    int heaps_per_block;            // Total number of heaps to go in the block
    int nheaps;                     // Number of heaps filled so far
    unsigned long long last_heap;   // Last heap counter written to block
};

/* Reset all counters */
void fake_reset_stats(struct fake_datablock_stats *d) {
    //d->npacket=0;
    //d->ndropped=0;
    d->last_heap=0;
}

/* Reset block params */
void fake_reset_block(struct fake_datablock_stats *d) {
    d->block_idx = -1;
    d->heap_idx = 0;
    fake_reset_stats(d);
}

/* Initialize block struct */
void fake_init_block(struct fake_datablock_stats *d, struct guppi_databuf *db, 
        size_t heap_size, int heaps_per_block) {
    d->db = db;
    d->heap_size = heap_size;
    d->heaps_per_block = heaps_per_block;
    fake_reset_block(d);
}

/* Update block header info, set filled status */
void fake_finalize_block(struct fake_datablock_stats *d) {
    char *header = guppi_databuf_header(d->db, d->block_idx);
    hputi4(header, "PKTIDX", d->heap_idx);
    hputi4(header, "PKTSIZE", d->heap_size);
    hputi4(header, "NPKT", 0);
    hputi4(header, "NDROP", 0);
    guppi_databuf_set_filled(d->db, d->block_idx);
}

/* Push all blocks down a level, losing the first one */
void fake_block_stack_push(struct fake_datablock_stats *d, int nblock) {
    int i;
    for (i=1; i<nblock; i++) 
        memcpy(&d[i-1], &d[i], sizeof(struct fake_datablock_stats));
}

/* Go to next block in set */
void fake_increment_block(struct fake_datablock_stats *d, unsigned long long next_heap_cntr) {
    d->block_idx = (d->block_idx + 1) % d->db->n_block;
    d->heap_idx = next_heap_cntr - (next_heap_cntr % d->heaps_per_block);
    fake_reset_stats(d);
}


/* Generate a fake heap and write it to shared mem block
 */
void write_fake_heap_to_block(struct fake_datablock_stats *d, int heap_cntr)
{
    int block_heap_idx = heap_cntr - d->heap_idx;
    int i;
    float spectrum_value;

    //Update the heap index first
    struct databuf_index* index = (struct databuf_index*)guppi_databuf_index(d->db, d->block_idx);
    index->num_heaps = index->num_heaps + 1;
    index->heap_size = d->heap_size;
    index->cpu_gpu_buf[block_heap_idx].heap_cntr = heap_cntr;
    index->cpu_gpu_buf[block_heap_idx].heap_valid = 1;

    //Create speed_heap at correct location in block
    struct freq_spead_heap* fake_heap;
    char *heap_addr = guppi_databuf_data(d->db, d->block_idx) + block_heap_idx*d->heap_size;
    fake_heap = (struct freq_spead_heap*)(heap_addr);
    float *payload = (float*)(heap_addr + sizeof(struct freq_spead_heap));

    //Populate necessary fields
    fake_heap->time_cntr = heap_cntr * 10;
    fake_heap->spectrum_cntr = heap_cntr;
    fake_heap->integ_size = 100;
    fake_heap->mode = 1;
    fake_heap->status_bits = 0;
    fake_heap->payload_data_off = 48;

    for(i = 0; i < SPEAD_SPECTRUM_SZ; i++)
    {
        spectrum_value = (float)i;
        memcpy(&(payload[i*4]), &spectrum_value, 4);
        memcpy(&(payload[i*4 + 1]), &spectrum_value, 4);
        memcpy(&(payload[i*4 + 2]), &spectrum_value, 4);
        memcpy(&(payload[i*4 + 3]), &spectrum_value, 4);
    }

    /* Update counters */
    d->nheaps++;
    d->last_heap = heap_cntr;
}


/* This thread is passed a single arg, pointer
 * to the guppi_udp_params struct.  This thread should 
 * be cancelled and restarted if any hardware params
 * change, as this potentially affects packet size, etc.
 */
void *guppi_fake_net_thread(void *_args) {

    /* Get arguments */
    struct guppi_thread_args *args = (struct guppi_thread_args *)_args;

    /* Set cpu affinity */
    cpu_set_t cpuset, cpuset_orig;
    sched_getaffinity(0, sizeof(cpu_set_t), &cpuset_orig);
    CPU_ZERO(&cpuset);
    //CPU_SET(2, &cpuset);
    CPU_SET(3, &cpuset);
    int rv = sched_setaffinity(0, sizeof(cpu_set_t), &cpuset);
    if (rv<0) { 
        guppi_error("guppi_fake_net_thread", "Error setting cpu affinity.");
        perror("sched_setaffinity");
    }

    /* Set priority */
    rv = setpriority(PRIO_PROCESS, 0, args->priority);
    if (rv<0) {
        guppi_error("guppi_fake_net_thread", "Error setting priority level.");
        perror("set_priority");
    }

    /* Attach to status shared mem area */
    struct guppi_status st;
    rv = guppi_status_attach(&st);
    if (rv!=GUPPI_OK) {
        guppi_error("guppi_fake_net_thread", 
                "Error attaching to status shared memory.");
        pthread_exit(NULL);
    }
    pthread_cleanup_push((void *)guppi_status_detach, &st);
    pthread_cleanup_push((void *)set_exit_status, &st);

    /* Init status, read info */
    guppi_status_lock_safe(&st);
    hputs(st.buf, STATUS_KEY, "init");
    guppi_status_unlock_safe(&st);

    /* Read in general parameters */
    struct guppi_params gp;
#if FITS_TYPE == PSRFITS
    struct psrfits pf;
    pf.sub.dat_freqs = NULL;
    pf.sub.dat_weights = NULL;
    pf.sub.dat_offsets = NULL;
    pf.sub.dat_scales = NULL;
#else
    struct sdfits pf;
#endif
    char status_buf[GUPPI_STATUS_SIZE];
    guppi_status_lock_safe(&st);
    memcpy(status_buf, st.buf, GUPPI_STATUS_SIZE);
    guppi_status_unlock_safe(&st);
    guppi_read_obs_params(status_buf, &gp, &pf);
#if FITS_TYPE == PSRFITS
    pthread_cleanup_push((void *)guppi_free_psrfits, &pf);
#else
    pthread_cleanup_push((void *)guppi_free_sdfits, &pf);
#endif

    /* Attach to databuf shared mem */
    struct guppi_databuf *db;
    db = guppi_databuf_attach(args->output_buffer); 
    if (db==NULL) {
        guppi_error("guppi_fake_net_thread",
                "Error attaching to databuf shared memory.");
        pthread_exit(NULL);
    }
    pthread_cleanup_push((void *)guppi_databuf_detach, db);

    /* Time parameters */
    int stt_imjd=0, stt_smjd=0;
    double stt_offs=0.0;

    /* Figure out size of data in each packet, number of packets
     * per block, etc.  Changing packet size during an obs is not
     * recommended.
     */
    int block_size;
    if (hgeti4(status_buf, "BLOCSIZE", &block_size)==0) {
            block_size = db->block_size;
            hputi4(status_buf, "BLOCSIZE", block_size);
    } else {
        if (block_size > db->block_size) {
            guppi_error("guppi_net_thread", "BLOCSIZE > databuf block_size");
            block_size = db->block_size;
            hputi4(status_buf, "BLOCSIZE", block_size);
        }
    }

    unsigned heaps_per_block = block_size / sizeof(struct freq_spead_heap);

    /* List of databuf blocks currently in use */
    unsigned i;
    const int nblock = 2;
    struct fake_datablock_stats blocks[nblock];
    for (i=0; i<nblock; i++) 
        fake_init_block(&blocks[i], db, sizeof(struct freq_spead_heap), heaps_per_block);

    /* Convenience names for first/last blocks in set */
    struct fake_datablock_stats *fblock, *lblock;
    fblock = &blocks[0];
    lblock = &blocks[nblock-1];

    /* Misc counters, etc */
    char *curdata=NULL, *curheader=NULL, *curindex=NULL;
    int first_time = 1;
    int heap_cntr = 0, next_block_heap_cntr = heaps_per_block;

    /* Main loop */
    unsigned force_new_block=0, waiting=-1;
    signal(SIGINT,cc);
    while (run) {

        /* Wait for data */
        struct timespec sleep_dur, rem_sleep_dur;
        sleep_dur.tv_sec = 0;
        sleep_dur.tv_nsec = 2e6;
        nanosleep(&sleep_dur, &rem_sleep_dur);
	
        /* Update status if needed */
        if (waiting!=0) {
            guppi_status_lock_safe(&st);
            hputs(st.buf, STATUS_KEY, "receiving");
            guppi_status_unlock_safe(&st);
            waiting=0;
        }

        /* Convert packet format if needed */
        if (first_time) 
        {
            first_time = 0;
            force_new_block=1;
        }
        else
            force_new_block=0; 

        /* Determine if we go to next block */
        if ((heap_cntr>=next_block_heap_cntr) || force_new_block) {

            printf("casper: going to next shared memory block\n");

            /* Update drop stats */
            guppi_status_lock_safe(&st);
            hputr8(st.buf, "DROPAVG", 0.0);
            hputr8(st.buf, "DROPTOT", 0.0);
            hputr8(st.buf, "DROPBLK", 0.0);
            guppi_status_unlock_safe(&st);
            
            /* Finalize first block, and push it off the list.
             * Then grab next available block.
             */
            if (fblock->block_idx>=0) fake_finalize_block(fblock);
            fake_block_stack_push(blocks, nblock);
            fake_increment_block(lblock, heap_cntr);
            curdata = guppi_databuf_data(db, lblock->block_idx);
            curheader = guppi_databuf_header(db, lblock->block_idx);
            curindex = guppi_databuf_index(db, lblock->block_idx);
            next_block_heap_cntr = lblock->heap_idx + heaps_per_block;

            /* If new obs started, reset total counters, get start
             * time.  Start time is rounded to nearest integer
             * second, with warning if we're off that by more
             * than 100ms.  Any current blocks on the stack
             * are also finalized/reset */
            if (force_new_block) {
            
                /* Get obs start time */
                get_current_mjd(&stt_imjd, &stt_smjd, &stt_offs);
                if (stt_offs>0.5) { stt_smjd+=1; stt_offs-=1.0; }
                stt_offs = 0.0;

                /* Flush any current buffers */
                for (i=0; i<nblock-1; i++) {
                    if (blocks[i].block_idx>=0) 
                        fake_finalize_block(&blocks[i]);
                    fake_reset_block(&blocks[i]);
                }

            }
            
            /* Read/update current status shared mem */
            guppi_status_lock_safe(&st);
            if (stt_imjd!=0) {
#if 1 
                hputi4(st.buf, "STT_IMJD", stt_imjd);
                hputi4(st.buf, "STT_SMJD", stt_smjd);
                hputr8(st.buf, "STT_OFFS", stt_offs);
#endif
                 hputi4(st.buf, "STTVALID", 1);
            } else {
                hputi4(st.buf, "STTVALID", 0);
            }
            memcpy(status_buf, st.buf, GUPPI_STATUS_SIZE);
            guppi_status_unlock_safe(&st);
 
            /* Wait for new block to be free, then clear it
             * if necessary and fill its header with new values.
             */
            while ((rv=guppi_databuf_wait_free(db, lblock->block_idx)) 
                    != GUPPI_OK) {
                if (rv==GUPPI_TIMEOUT) {
                    waiting=1;
                    guppi_status_lock_safe(&st);
                    hputs(st.buf, STATUS_KEY, "blocked");
                    guppi_status_unlock_safe(&st);
                    continue;
                } else {
                    guppi_error("guppi_fake_net_thread", 
                            "error waiting for free databuf");
                    run=0;
                    pthread_exit(NULL);
                    break;
                }
            }
            memcpy(curheader, status_buf, GUPPI_STATUS_SIZE);
            memset(curdata, 0, block_size);
            memset(curindex, 0, db->index_size);
        }

        /*Write fake data to block */ 
        write_fake_heap_to_block(lblock, heap_cntr);
        heap_cntr++;

        /* Will exit if thread has been cancelled */
        pthread_testcancel();
    }

    pthread_exit(NULL);

    /* Have to close all push's */
    pthread_cleanup_pop(0); /* Closes set_exit_status */
    pthread_cleanup_pop(0); /* Closes guppi_free_psrfits */
    pthread_cleanup_pop(0); /* Closes guppi_status_detach */
    pthread_cleanup_pop(0); /* Closes guppi_databuf_detach */
}
