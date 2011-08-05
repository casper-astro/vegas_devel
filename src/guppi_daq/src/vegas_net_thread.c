/* vegas_net_thread.c
 *
 * Routine to read packets from network and put them
 * into shared memory blocks.
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

#define STATUS_KEY "NETSTAT"  /* Define before guppi_threads.h */
#include "guppi_threads.h"
#include "guppi_defines.h"

/* This file will only compile if the FITS_TYPE is set to SDFITS */
#if FITS_TYPE != SDFITS
#error "FITS_TYPE not set to SDFITS."
#endif

#include "sdfits.h"
#include "spead_heap.h"

// Read a status buffer all of the key observation paramters
extern void guppi_read_obs_params(char *buf, 
                                  struct guppi_params *g, 
                                  struct sdfits *p);

/* Structs/functions to more easily deal with multiple 
 * active blocks being filled
 */
struct datablock_stats {
    struct guppi_databuf *db;       // Pointer to overall shared mem databuf
    int block_idx;                  // Block index number in databuf
    unsigned int heap_idx;          // Index of first heap in block
    size_t heap_size;               // Size of each heap
    size_t spead_hdr_size;          // Size of each SPEAD header
    int heaps_per_block;            // Total number of heaps to go in the block
    int nheaps;                     // Number of heaps filled so far
    int pkts_dropped;               // Number of dropped packets so far
    unsigned int last_heap;         // Last heap counter written to block
};

/* Reset all counters */
void reset_stats(struct datablock_stats *d) {
    d->nheaps=0;
    d->pkts_dropped=0;
    d->last_heap=0;
}

/* Reset block params */
void reset_block(struct datablock_stats *d) {
    d->block_idx = -1;
    d->heap_idx = 0;

    reset_stats(d);
}

/* Initialize block struct */
void init_block(struct datablock_stats *d, struct guppi_databuf *db, 
        size_t heap_size, size_t spead_hdr_size, int heaps_per_block) {
    d->db = db;
    d->heap_size = heap_size;
    d->spead_hdr_size = spead_hdr_size;
    d->heaps_per_block = heaps_per_block;
    reset_block(d);
}

/* Update block header info, set filled status */
void finalize_block(struct datablock_stats *d) {
    char *header = guppi_databuf_header(d->db, d->block_idx);
    hputi4(header, "HEAPIDX", d->heap_idx);
    hputi4(header, "HEAPSIZE", d->heap_size);
    hputi4(header, "NHEAPS", d->nheaps);
    hputi4(header, "NDROP", d->pkts_dropped);

    struct databuf_index* index = (struct databuf_index*)
                                guppi_databuf_index(d->db, d->block_idx);
    index->num_heaps = d->nheaps;
    index->heap_size = d->heap_size;

    guppi_databuf_set_filled(d->db, d->block_idx);
}

/* Push all blocks down a level, losing the first one */
void block_stack_push(struct datablock_stats *d, int nblock) {
    int i;
    for (i=1; i<nblock; i++) 
        memcpy(&d[i-1], &d[i], sizeof(struct datablock_stats));
}

/* Go to next block in set */
void increment_block(struct datablock_stats *d, unsigned int next_heap_cntr)
{
    d->block_idx = (d->block_idx + 1) % d->db->n_block;
    d->heap_idx = next_heap_cntr;
    reset_stats(d);
}

/* Check whether a certain heap counter belongs in the data block */
int block_heap_check(struct datablock_stats *d, unsigned int heap_cntr) {
    if (heap_cntr < d->heap_idx)
        return(-1);
    else if (heap_cntr >= d->heap_idx + d->heaps_per_block)
        return(1);
    else return(0);
}


/*
 *  Write a SPEAD packet into the datablock.  Also zeroes out any dropped packets.
 */
unsigned int prev_heap_cntr;
unsigned int prev_heap_offset;
char pkts_dropped_in_heap;

void write_spead_packet_to_block(struct datablock_stats *d, struct guppi_udp_packet *p,
                                unsigned int heap_cntr, unsigned int heap_offset,
                                unsigned int pkts_per_heap, char bw_mode[])
{
    int block_heap_idx;
    char *spead_header_addr, *spead_payload_addr;
    double mjd;

    /*Determine packet's address within block */
    block_heap_idx = heap_cntr - d->heap_idx;

    spead_header_addr = guppi_databuf_data(d->db, d->block_idx) + 
                block_heap_idx * d->spead_hdr_size;
    spead_payload_addr = guppi_databuf_data(d->db, d->block_idx) +
                MAX_HEAPS_PER_BLK * d->spead_hdr_size +
                block_heap_idx * (d->heap_size - d->spead_hdr_size) +
                (heap_offset > 0 ? heap_offset - d->spead_hdr_size : 0);

    /* Copy packet to address, while reversing the byte ordering */
    guppi_spead_packet_copy(p, spead_header_addr, spead_payload_addr, bw_mode);

    /*Update block statistics */
    d->nheaps = block_heap_idx + 1;
    d->last_heap = heap_cntr;

    //Determine if we dropped any packets while in same heap
    if( guppi_spead_packet_seq_num(heap_cntr, heap_offset, pkts_per_heap) - 
        guppi_spead_packet_seq_num(prev_heap_cntr, prev_heap_offset, pkts_per_heap) > 1 &&
        prev_heap_cntr == heap_cntr)
    {
        pkts_dropped_in_heap = 1;
    }

    struct databuf_index* index = (struct databuf_index*)
                            guppi_databuf_index(d->db, d->block_idx);

    //If start of new heap, write it to index
    if(heap_cntr != prev_heap_cntr)
    {
        index->cpu_gpu_buf[block_heap_idx].heap_cntr = heap_cntr;
        index->cpu_gpu_buf[block_heap_idx].heap_valid = 0;

        if(heap_offset == 0)
            pkts_dropped_in_heap = 0;
        else
            pkts_dropped_in_heap = 1;
    }

    //If this is the last packet of the heap, write valid bit and MJD to index
    if(heap_offset + PAYLOAD_SIZE + 6*8 >= d->heap_size)
    {
		index->cpu_gpu_buf[block_heap_idx].heap_valid = 1 - pkts_dropped_in_heap;
        get_current_mjd_double(&mjd);
        index->cpu_gpu_buf[block_heap_idx].heap_rcvd_mjd = mjd;
    }

    prev_heap_cntr = heap_cntr;
    prev_heap_offset = heap_offset;
}


/* This thread is passed a single arg, pointer
 * to the guppi_udp_params struct.  This thread should 
 * be cancelled and restarted if any hardware params
 * change, as this potentially affects packet size, etc.
 */
void *guppi_net_thread(void *_args) {

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
        guppi_error("guppi_net_thread", "Error setting cpu affinity.");
        perror("sched_setaffinity");
    }

    /* Set priority */
    rv = setpriority(PRIO_PROCESS, 0, args->priority);
    if (rv<0) {
        guppi_error("guppi_net_thread", "Error setting priority level.");
        perror("set_priority");
    }

    /* Attach to status shared mem area */
    struct guppi_status st;
    rv = guppi_status_attach(&st);
    if (rv!=GUPPI_OK) {
        guppi_error("guppi_net_thread", 
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
    struct sdfits pf;
    char status_buf[GUPPI_STATUS_SIZE];
    guppi_status_lock_safe(&st);
    memcpy(status_buf, st.buf, GUPPI_STATUS_SIZE);
    guppi_status_unlock_safe(&st);
    guppi_read_obs_params(status_buf, &gp, &pf);
    pthread_cleanup_push((void *)guppi_free_sdfits, &pf);

    /* Read network params */
    struct guppi_udp_params up;
    guppi_read_net_params(status_buf, &up);

    /* Attach to databuf shared mem */
    struct guppi_databuf *db;
    db = guppi_databuf_attach(args->output_buffer); 
    if (db==NULL) {
        guppi_error("guppi_net_thread",
                "Error attaching to databuf shared memory.");
        pthread_exit(NULL);
    }
    pthread_cleanup_push((void *)guppi_databuf_detach, db);

    /* Set up UDP socket */
    rv = guppi_udp_init(&up);
    if (rv!=GUPPI_OK) {
        guppi_error("guppi_net_thread",
                "Error opening UDP socket.");
        pthread_exit(NULL);
    }
    pthread_cleanup_push((void *)guppi_udp_close, &up);

    /* Time parameters */
    double meas_stt_mjd=0.0;
    double meas_stt_offs=0.0;

    /* See which packet format to use */
    int nchan=0, npol=0;
    nchan = pf.hdr.nchan;
    npol = pf.hdr.npol;

    /* Figure out size of data in each packet, number of packets
     * per block, etc.  Changing packet size during an obs is not
     * recommended.
     */
    int block_size;
    struct guppi_udp_packet p;
    size_t heap_size, spead_hdr_size;
    unsigned int heaps_per_block, packets_per_heap; 
    char bw_mode[16];

    if (hgets(status_buf, "BW_MODE", 16, bw_mode))
    {
        if(strncmp(bw_mode, "high", 4) == 0)
        {
            heap_size = sizeof(struct freq_spead_heap) + nchan*4*sizeof(float);
            spead_hdr_size = sizeof(struct freq_spead_heap);
            packets_per_heap = nchan*4*sizeof(float) / PAYLOAD_SIZE;
        }
        else if(strncmp(bw_mode, "low", 3) == 0)
        {
            heap_size = sizeof(struct time_spead_heap) + PAYLOAD_SIZE;
            spead_hdr_size = sizeof(struct time_spead_heap);
            packets_per_heap = 1;
        }
        else
            guppi_error("guppi_net_thread", "Unsupported bandwidth mode");
    }
    else
        guppi_error("guppi_net_thread", "BW_MODE not set");

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
    heaps_per_block = block_size / heap_size;

    /* List of databuf blocks currently in use */
    unsigned i;
    const int nblock = 2;
    struct datablock_stats blocks[nblock];
    for (i=0; i<nblock; i++) 
        init_block(&blocks[i], db, heap_size, spead_hdr_size, heaps_per_block);

    /* Convenience names for first/last blocks in set */
    struct datablock_stats *fblock, *lblock;
    fblock = &blocks[0];
    lblock = &blocks[nblock-1];

    /* Misc counters, etc */
    char *curdata=NULL, *curheader=NULL, *curindex=NULL;
    unsigned int heap_cntr=0, last_heap_cntr=2048, nextblock_heap_cntr=0;
    unsigned int heap_offset;
    unsigned int seq_num=0, last_seq_num=1050;
    int heap_cntr_diff, seq_num_diff;
    unsigned int obs_started = 0;
    unsigned long long npacket_total, npacket_this_block=0, ndropped_total;
    double drop_frac_avg=0.0;
    const double drop_lpf = 0.25;
    prev_heap_cntr = 0;
    prev_heap_offset = 0;

    /* Main loop */
    unsigned force_new_block=0, waiting=-1;
    signal(SIGINT,cc);
    while (run) {

        /* Wait for data */
        rv = guppi_udp_wait(&up);
        if (rv!=GUPPI_OK) {
            if (rv==GUPPI_TIMEOUT) { 
                /* Set "waiting" flag */
                if (waiting!=1) {
                    guppi_status_lock_safe(&st);
                    hputs(st.buf, STATUS_KEY, "waiting");
                    guppi_status_unlock_safe(&st);
                    waiting=1;
                }
                continue; 
            } else {
                guppi_error("guppi_net_thread", 
                        "guppi_udp_wait returned error");
                perror("guppi_udp_wait");
                pthread_exit(NULL);
            }
        }
	
        /* Read packet */
        rv = guppi_udp_recv(&up, &p);
        if (rv!=GUPPI_OK) {
            if (rv==GUPPI_ERR_PACKET) {
                /* Unexpected packet size, ignore? */
printf("Error: incorrect pkt size\n");
                continue; 
            } else {
                guppi_error("guppi_net_thread", 
                        "guppi_udp_recv returned error");
                perror("guppi_udp_recv");
                pthread_exit(NULL);
            }
        }
	
        /* Update status if needed */
        if (waiting!=0) {
            guppi_status_lock_safe(&st);
            hputs(st.buf, STATUS_KEY, "receiving");
            guppi_status_unlock_safe(&st);
            waiting=0;
        }

        /* Check seq num diff */
        heap_cntr = guppi_spead_packet_heap_cntr(&p);
        heap_offset = guppi_spead_packet_heap_offset(&p);
        seq_num = guppi_spead_packet_seq_num(heap_cntr, heap_offset, packets_per_heap);

        heap_cntr_diff = heap_cntr - last_heap_cntr;
        seq_num_diff = (int)(seq_num - last_seq_num);
        
        if (seq_num_diff<=0) { 

             if (seq_num_diff<-1024)
            {
                force_new_block=1;
                obs_started = 1;
            }
            else if (seq_num_diff==0) {
                char msg[256];
                sprintf(msg, "Received duplicate packet (seq_num=%d)", seq_num);
                guppi_warn("guppi_net_thread", msg);
            }
            else  {
printf("Error: out of order packet. Diff = %d\n", seq_num_diff);
                continue;   /* No going backwards */
            }
        } else { 
            force_new_block=0; 
            npacket_total += seq_num_diff;
            ndropped_total += seq_num_diff - 1;
            npacket_this_block += seq_num_diff;
            fblock->pkts_dropped += seq_num_diff - 1;
if(seq_num_diff > 1) printf("Error: missing packet: seq_num_diff = %d\n", seq_num_diff);
        }
        last_seq_num = seq_num;
        last_heap_cntr = heap_cntr;

        /* If obs has not started, ignore this packet */
        if(!obs_started)
        {
            fblock->pkts_dropped = 0;
            npacket_total = 0;
            ndropped_total = 0;
            npacket_this_block = 0;

            continue;
        }

        /* Determine if we go to next block */
        if (heap_cntr>=nextblock_heap_cntr || force_new_block)
        {
            printf("Debug: vegas_net_thread writing to next output block\n");

            /* Update drop stats */
            if (npacket_this_block > 0)  
                drop_frac_avg = (1.0-drop_lpf)*drop_frac_avg 
                    + drop_lpf *
                    (double)fblock->pkts_dropped / (double)npacket_this_block;

            guppi_status_lock_safe(&st);
            hputi8(st.buf, "NPKT", npacket_total);
            hputi8(st.buf, "NDROP", ndropped_total);
            hputr8(st.buf, "DROPAVG", drop_frac_avg);
            hputr8(st.buf, "DROPTOT", 
                    npacket_total ? 
                    (double)ndropped_total/(double)npacket_total 
                    : 0.0);
            hputi4(st.buf, "NETBLKOU", fblock->block_idx);
            guppi_status_unlock_safe(&st);
            
            /* Finalize first block, and push it off the list.
             * Then grab next available block.
             */
            if (fblock->block_idx>=0) finalize_block(fblock);
            block_stack_push(blocks, nblock);
            increment_block(lblock, heap_cntr);
            curdata = guppi_databuf_data(db, lblock->block_idx);
            curheader = guppi_databuf_header(db, lblock->block_idx);
            curindex = guppi_databuf_index(db, lblock->block_idx);
            nextblock_heap_cntr = lblock->heap_idx + heaps_per_block;
            npacket_this_block = 0;

            /* If new obs started, reset total counters, get start
             * time.  Start time is rounded to nearest integer
             * second, with warning if we're off that by more
             * than 100ms.  Any current blocks on the stack
             * are also finalized/reset */
            if (force_new_block) {
            
                /* Reset stats */
                npacket_total=0;
                ndropped_total=0;
                npacket_this_block = 0;

                /* Get obs start time */
                get_current_mjd_double(&meas_stt_mjd);
                meas_stt_offs = meas_stt_mjd*24*60*60 - floor(meas_stt_mjd*24*60*60);

                if(meas_stt_offs > 0.1 && meas_stt_offs < 0.9)
                { 
                    char msg[256];
                    sprintf(msg, 
                            "Second fraction = %3.1f ms > +/-100 ms",
                            meas_stt_offs*1e3);
                    guppi_warn("guppi_net_thread", msg);
                }

                guppi_status_lock_safe(&st);
                hputnr8(st.buf, "M_STTMJD", 8, meas_stt_mjd);
                hputr8(st.buf, "M_STTOFF", meas_stt_offs);
                guppi_status_unlock_safe(&st);

                /* Warn if 1st packet number is not zero */
                if (seq_num!=0) {
                    char msg[256];
                    sprintf(msg, "First packet number is not 0 (seq_num=%d)", seq_num);
                    guppi_warn("guppi_net_thread", msg);
                }
            
                /* Flush any current buffers */
                for (i=0; i<nblock-1; i++) {
                    if (blocks[i].block_idx>=0) 
                        finalize_block(&blocks[i]);
                    reset_block(&blocks[i]);
                }

            }
            
            /* Read current status shared mem */
            guppi_status_lock_safe(&st);
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
                    guppi_error("guppi_net_thread", 
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

        /* Copy packet into any blocks where it belongs.
         * The "write packets" functions also update drop stats 
         * for blocks, etc.
         */
        for (i=0; i<nblock; i++)
        {
            if ((blocks[i].block_idx>=0) && (block_heap_check(&blocks[i],heap_cntr)==0))
            {
                write_spead_packet_to_block(&blocks[i], &p, heap_cntr,
                                heap_offset, packets_per_heap, bw_mode);
            }
        }

        /* Will exit if thread has been cancelled */
        pthread_testcancel();
    }

    pthread_exit(NULL);

    /* Have to close all push's */
    pthread_cleanup_pop(0); /* Closes push(guppi_udp_close) */
    pthread_cleanup_pop(0); /* Closes set_exit_status */
    pthread_cleanup_pop(0); /* Closes guppi_free_psrfits */
    pthread_cleanup_pop(0); /* Closes guppi_status_detach */
    pthread_cleanup_pop(0); /* Closes guppi_databuf_detach */
}
