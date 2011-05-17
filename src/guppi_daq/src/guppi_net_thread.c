/* guppi_net_thread.c
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
#include "psrfits.h"
#include "guppi_error.h"
#include "guppi_status.h"
#include "guppi_databuf.h"
#include "guppi_udp.h"
#include "guppi_time.h"

#define STATUS_KEY "NETSTAT"  /* Define before guppi_threads.h */
#include "guppi_threads.h"
#include "guppi_defines.h"

// Read a status buffer all of the key observation paramters
extern void guppi_read_obs_params(char *buf, 
                                  struct guppi_params *g, 
                                  struct psrfits *p);

/* It's easier to just make these global ... */
static unsigned long long npacket_total=0, ndropped_total=0, nbogus_total=0;

/* Structs/functions to more easily deal with multiple 
 * active blocks being filled
 */
struct datablock_stats {
    struct guppi_databuf *db;      // Pointer to overall shared mem databuf
    int block_idx;                 // Block index number in databuf
    unsigned long long packet_idx; // Index of first packet number in block
    size_t packet_data_size;       // Data size of each packet
    int packets_per_block;         // Total number of packets to go in the block
    int overlap_packets;           // Overlap between blocks in packets
    int npacket;                   // Number of packets filled so far
    int ndropped;                  // Number of dropped packets so far
    unsigned long long last_pkt;   // Last packet seq number written to block
};

/* Reset all counters */
void reset_stats(struct datablock_stats *d) {
    d->npacket=0;
    d->ndropped=0;
    d->last_pkt=0;
}

/* Reset block params */
void reset_block(struct datablock_stats *d) {
    d->block_idx = -1;
    d->packet_idx = 0;
    reset_stats(d);
}

/* Initialize block struct */
void init_block(struct datablock_stats *d, struct guppi_databuf *db, 
        size_t packet_data_size, int packets_per_block, int overlap_packets) {
    d->db = db;
    d->packet_data_size = packet_data_size;
    d->packets_per_block = packets_per_block;
    d->overlap_packets = overlap_packets;
    reset_block(d);
}

/* Update block header info, set filled status */
void finalize_block(struct datablock_stats *d) {
    char *header = guppi_databuf_header(d->db, d->block_idx);
    hputi4(header, "PKTIDX", d->packet_idx);
    hputi4(header, "PKTSIZE", d->packet_data_size);
    hputi4(header, "NPKT", d->npacket);
    hputi4(header, "NDROP", d->ndropped);
    guppi_databuf_set_filled(d->db, d->block_idx);
}

/* Push all blocks down a level, losing the first one */
void block_stack_push(struct datablock_stats *d, int nblock) {
    int i;
    for (i=1; i<nblock; i++) 
        memcpy(&d[i-1], &d[i], sizeof(struct datablock_stats));
}

/* Go to next block in set */
void increment_block(struct datablock_stats *d, 
        unsigned long long next_seq_num) {
    d->block_idx = (d->block_idx + 1) % d->db->n_block;
    d->packet_idx = next_seq_num - (next_seq_num 
            % (d->packets_per_block - d->overlap_packets));
    reset_stats(d);
    // TODO: wait for block free here?
}

/* Check whether a certain seq num belongs in the data block */
int block_packet_check(struct datablock_stats *d, 
        unsigned long long seq_num) {
    if (seq_num < d->packet_idx) return(-1);
    else if (seq_num >= d->packet_idx + d->packets_per_block) return(1);
    else return(0);
}

/* Write a search mode (filterbank) style packet into the
 * datablock.  Also zeroes out any dropped packets.
 */
void write_search_packet_to_block(struct datablock_stats *d, 
        struct guppi_udp_packet *p) {
    const unsigned long long seq_num = guppi_udp_packet_seq_num(p);
    int next_pos = seq_num - d->packet_idx;
    int cur_pos=0;
    if (d->last_pkt > d->packet_idx) cur_pos = d->last_pkt - d->packet_idx + 1;
    char *dataptr = guppi_databuf_data(d->db, d->block_idx) 
        + cur_pos*d->packet_data_size;
    for (; cur_pos<next_pos; cur_pos++) {
        memset(dataptr, 0, d->packet_data_size);
        dataptr += d->packet_data_size;
        d->npacket++;
        d->ndropped++;
    }
    guppi_udp_packet_data_copy(dataptr, p);
    d->last_pkt = seq_num;
    //d->packet_idx++; // XXX I think this is wrong..
    d->npacket++;
}

/* Write a baseband mode packet into the block.  Includes a 
 * corner-turn (aka transpose) of dimension nchan.
 */
void write_baseband_packet_to_block(struct datablock_stats *d, 
        struct guppi_udp_packet *p, int nchan) {

    const unsigned long long seq_num = guppi_udp_packet_seq_num(p);
    int block_pkt_idx = seq_num - d->packet_idx;

#ifdef NEW_GBT
    guppi_udp_packet_data_copy(guppi_databuf_data(d->db, 0), p);

#else

    guppi_udp_packet_data_copy_transpose(
            guppi_databuf_data(d->db, d->block_idx),
            nchan, block_pkt_idx, 
            d->packets_per_block, p);
#endif

    /* Consider any skipped packets to have been dropped,
     * update counters.
     */
    if (d->last_pkt < d->packet_idx) d->last_pkt = d->packet_idx;

    if (seq_num == d->last_pkt) {
        d->npacket++;
    } else {
        d->npacket += seq_num - d->last_pkt;
        d->ndropped += seq_num - d->last_pkt - 1;
    }

    d->last_pkt = seq_num;
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
    struct psrfits pf;
    pf.sub.dat_freqs = NULL;
    pf.sub.dat_weights = NULL;
    pf.sub.dat_offsets = NULL;
    pf.sub.dat_scales = NULL;
    char status_buf[GUPPI_STATUS_SIZE];
    guppi_status_lock_safe(&st);
    memcpy(status_buf, st.buf, GUPPI_STATUS_SIZE);
    guppi_status_unlock_safe(&st);
    guppi_read_obs_params(status_buf, &gp, &pf);
    pthread_cleanup_push((void *)guppi_free_psrfits, &pf);

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
    int stt_imjd=0, stt_smjd=0;
    double stt_offs=0.0;

    /* See which packet format to use */
    int use_parkes_packets=0, baseband_packets=1;
    int nchan=0, npol=0, acclen=0;
    nchan = pf.hdr.nchan;
    npol = pf.hdr.npol;
    if (strncmp(up.packet_format, "PARKES", 6)==0) { use_parkes_packets=1; }
    if (use_parkes_packets) {
        printf("guppi_net_thread: Using Parkes UDP packet format.\n");
        acclen = gp.decimation_factor;
        if (acclen==0) { 
            guppi_error("guppi_net_thread", 
                    "ACC_LEN must be set to use Parkes format");
            pthread_exit(NULL);
        }
    }

    /* Figure out size of data in each packet, number of packets
     * per block, etc.  Changing packet size during an obs is not
     * recommended.
     */
    int block_size;
    struct guppi_udp_packet p;
    size_t packet_data_size = guppi_udp_packet_datasize(up.packet_size); 
    if (use_parkes_packets) 
        packet_data_size = parkes_udp_packet_datasize(up.packet_size);
    unsigned packets_per_block; 
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
    packets_per_block = block_size / packet_data_size;

    /* If we're in baseband mode, figure out how much to overlap
     * the data blocks.
     */
    int overlap_packets=0;
    if (baseband_packets) {
        if (hgeti4(status_buf, "OVERLAP", &overlap_packets)==0) {
            overlap_packets = 0; // Default to no overlap
        } else {
            // XXX This is only true for 8-bit, 2-pol data:
            int samples_per_packet = packet_data_size / nchan / (size_t)4;
            if (overlap_packets % samples_per_packet) {
                guppi_error("guppi_net_thread", 
                        "Overlap is not an integer number of packets");
                overlap_packets = (overlap_packets/samples_per_packet+1);
                hputi4(status_buf, "OVERLAP", 
                        overlap_packets*samples_per_packet);
            } else {
                overlap_packets = overlap_packets/samples_per_packet;
            }
        }
    }

    /* List of databuf blocks currently in use */
    unsigned i;
    const int nblock = 2;
    struct datablock_stats blocks[nblock];
    for (i=0; i<nblock; i++) 
        init_block(&blocks[i], db, packet_data_size, packets_per_block, 
                overlap_packets);

    /* Convenience names for first/last blocks in set */
    struct datablock_stats *fblock, *lblock;
    fblock = &blocks[0];
    lblock = &blocks[nblock-1];

    /* Misc counters, etc */
    char *curdata=NULL, *curheader=NULL;
    unsigned long long seq_num=0, last_seq_num=2048, nextblock_seq_num=0;
    long long seq_num_diff;
    double drop_frac_avg=0.0;
    const double drop_lpf = 0.25;

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
                nbogus_total++;
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

        /* Convert packet format if needed */
        if (use_parkes_packets) 
            parkes_to_guppi(&p, acclen, npol, nchan);

        /* Check seq num diff */

#if !defined(SPEAD)

       seq_num = guppi_udp_packet_seq_num(&p);

#else

        /* Check for missing SPEAD packets by checking BOTH the heap counter and 
         * heap pointer. If the heap counter is the same as the last packet's heap
         * counter, then the heap pointer must be checked that it incremented by the
         * previous packet's payload size. */
        //TODO
        if(last_seq_num == 2048 && seq_num == 0)
            seq_num = 0;
        else
            seq_num = last_seq_num + 1;

#endif

        seq_num_diff = seq_num - last_seq_num;

        if (seq_num_diff<=0) { 
            if (seq_num_diff<-1024) { force_new_block=1; }
            else if (seq_num_diff==0) {
                char msg[256];
                sprintf(msg, "Received duplicate packet (seq_num=%lld)", 
                        seq_num);
                guppi_warn("guppi_net_thread", msg);
            }
            else  { continue; } /* No going backwards */
        } else { 
            force_new_block=0; 
            npacket_total += seq_num_diff;
            ndropped_total += seq_num_diff - 1;
        }
        last_seq_num = seq_num;

        /* Determine if we go to next block */
        if ((seq_num>=nextblock_seq_num) || force_new_block) {
            printf("casper: going to next shared memory block\n");
            /* Update drop stats */
            if (fblock->npacket)  
                drop_frac_avg = (1.0-drop_lpf)*drop_frac_avg 
                    + drop_lpf * 
                    (double)fblock->ndropped / 
                    (double)fblock->npacket;

            guppi_status_lock_safe(&st);
            hputr8(st.buf, "DROPAVG", drop_frac_avg);
            hputr8(st.buf, "DROPTOT", 
                    npacket_total ? 
                    (double)ndropped_total/(double)npacket_total 
                    : 0.0);
            hputr8(st.buf, "DROPBLK", 
                    fblock->npacket ? 
                    (double)fblock->ndropped/(double)fblock->npacket
                    : 0.0);
            guppi_status_unlock_safe(&st);
            
            /* Finalize first block, and push it off the list.
             * Then grab next available block.
             */
            if (fblock->block_idx>=0) finalize_block(fblock);
            block_stack_push(blocks, nblock);
            increment_block(lblock, seq_num);
            curdata = guppi_databuf_data(db, lblock->block_idx);
            curheader = guppi_databuf_header(db, lblock->block_idx);
            nextblock_seq_num = lblock->packet_idx 
                + packets_per_block - overlap_packets;

            /* If new obs started, reset total counters, get start
             * time.  Start time is rounded to nearest integer
             * second, with warning if we're off that by more
             * than 100ms.  Any current blocks on the stack
             * are also finalized/reset */
            if (force_new_block) {
            
                /* Reset stats */
                npacket_total=0;
                ndropped_total=0;
                nbogus_total=0;

                /* Get obs start time */
                get_current_mjd(&stt_imjd, &stt_smjd, &stt_offs);
                if (stt_offs>0.5) { stt_smjd+=1; stt_offs-=1.0; }
                if (fabs(stt_offs)>0.1) { 
                    char msg[256];
                    sprintf(msg, 
                            "Second fraction = %3.1f ms > +/-100 ms",
                            stt_offs*1e3);
                    guppi_warn("guppi_net_thread", msg);
                }
                stt_offs = 0.0;

                /* Warn if 1st packet number is not zero */
                if (seq_num!=0) {
                    char msg[256];
                    sprintf(msg, "First packet number is not 0 (seq_num=%lld)",
                            seq_num);
                    guppi_warn("guppi_net_thread", msg);
                }
            
                /* Flush any current buffers */
                for (i=0; i<nblock-1; i++) {
                    if (blocks[i].block_idx>=0) 
                        finalize_block(&blocks[i]);
                    reset_block(&blocks[i]);
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
                // XXX REMOVE WHEN DONE TESTING XXX
                //hputi4(st.buf, "STT_IMJD", 54552);
                //hputi4(st.buf, "STT_SMJD", 21447);
                //hputr8(st.buf, "STT_OFFS", 0.0);
                // 1937:
                //hputi4(st.buf, "STT_IMJD", 55049);
                //hputi4(st.buf, "STT_SMJD", 83701);
                //hputr8(st.buf, "STT_OFFS", 0.0);
                // XXX REMOVE WHEN DONE TESTING XXX
                hputi4(st.buf, "STTVALID", 1);
            } else {
                hputi4(st.buf, "STTVALID", 0);
            }
            memcpy(status_buf, st.buf, GUPPI_STATUS_SIZE);
            guppi_status_unlock_safe(&st);
            
            /* block size possibly changed on new obs */
            /* TODO: what about overlap...
             * Also, should this even be allowed ?
             */
            if (force_new_block) {
                if (hgeti4(status_buf, "BLOCSIZE", &block_size)==0) {
                        block_size = db->block_size;
                } else {
                    if (block_size > db->block_size) {
                        guppi_error("guppi_net_thread", 
                                "BLOCSIZE > databuf block_size");
                        block_size = db->block_size;
                    }
                }
                packets_per_block = block_size / packet_data_size;
            }
            hputi4(status_buf, "BLOCSIZE", block_size);

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
            //if (baseband_packets) { memset(curdata, 0, block_size); }
            if (1) { memset(curdata, 0, block_size); }

        }

        /* Copy packet into any blocks where it belongs.
         * The "write packets" functions also update drop stats 
         * for blocks, etc.
         */
        for (i=0; i<nblock; i++) {
            if ((blocks[i].block_idx>=0) 
                    && (block_packet_check(&blocks[i],seq_num)==0)) {
                if (baseband_packets) 
                    write_baseband_packet_to_block(&blocks[i], &p, nchan);
                 else
                    write_search_packet_to_block(&blocks[i], &p);
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
