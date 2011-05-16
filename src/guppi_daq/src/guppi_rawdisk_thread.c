/* guppi_rawdisk_thread.c
 *
 * Write databuf blocks out to disk.
 */

#define _GNU_SOURCE 1
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>
#include <pthread.h>
#include <signal.h>
#include <sched.h>
#include <sys/time.h>
#include <sys/resource.h>
#include <sys/types.h>

#include "fitshead.h"
#include "psrfits.h"
#include "quantization.h"
#include "guppi_error.h"
#include "guppi_status.h"
#include "guppi_databuf.h"

#define STATUS_KEY "DISKSTAT"
#include "guppi_threads.h"

// Read a status buffer all of the key observation paramters
extern void guppi_read_obs_params(char *buf, 
                                  struct guppi_params *g, 
                                  struct psrfits *p);

/* Parse info from buffer into param struct */
extern void guppi_read_subint_params(char *buf, 
                                     struct guppi_params *g,
                                     struct psrfits *p);

int safe_fclose(FILE *f) {
    if (f==NULL) return 0;
    sync();
    return fclose(f);
}

void guppi_rawdisk_thread(void *_args) {

    /* Set cpu affinity */
    cpu_set_t cpuset, cpuset_orig;
    sched_getaffinity(0, sizeof(cpu_set_t), &cpuset_orig);
    CPU_ZERO(&cpuset);
    CPU_SET(1, &cpuset);
    int rv = sched_setaffinity(0, sizeof(cpu_set_t), &cpuset);
    if (rv<0) { 
        guppi_error("guppi_rawdisk_thread", "Error setting cpu affinity.");
        perror("sched_setaffinity");
    }

    /* Get args */
    struct guppi_thread_args *args = (struct guppi_thread_args *)_args;

    /* Set priority */
    rv = setpriority(PRIO_PROCESS, 0, 0);
    if (rv<0) {
        guppi_error("guppi_rawdisk_thread", "Error setting priority level.");
        perror("set_priority");
    }

    /* Attach to status shared mem area */
    struct guppi_status st;
    rv = guppi_status_attach(&st);
    if (rv!=GUPPI_OK) {
        guppi_error("guppi_rawdisk_thread", 
                "Error attaching to status shared memory.");
        pthread_exit(NULL);
    }
    pthread_cleanup_push((void *)guppi_status_detach, &st);
    pthread_cleanup_push((void *)set_exit_status, &st);

    /* Init status */
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
    pthread_cleanup_push((void *)guppi_free_psrfits, &pf);

    /* Attach to databuf shared mem */
    struct guppi_databuf *db;
    db = guppi_databuf_attach(args->input_buffer);
    if (db==NULL) {
        guppi_error("guppi_rawdisk_thread",
                "Error attaching to databuf shared memory.");
        pthread_exit(NULL);
    }
    pthread_cleanup_push((void *)guppi_databuf_detach, db);

    /* Init output file */
    FILE *fraw = NULL;
    pthread_cleanup_push((void *)safe_fclose, fraw);

    /* Pointers for quantization params */
    double *mean = NULL;
    double *std = NULL;
    printf("casper: raw disk thread created and running\n");
    /* Loop */
    int packetidx=0, npacket=0, ndrop=0, packetsize=0, blocksize=0;
    int orig_blocksize=0;
    int curblock=0;
    int block_count=0, blocks_per_file=128, filenum=0;
    int got_packet_0=0, first=1;
    int requantize = 0;
    char *ptr, *hend;
    signal(SIGINT,cc);
    while (run) {

        /* Note waiting status */
        guppi_status_lock_safe(&st);
        hputs(st.buf, STATUS_KEY, "waiting");
        guppi_status_unlock_safe(&st);

        /* Wait for buf to have data */
        rv = guppi_databuf_wait_filled(db, curblock);
        if (rv!=0) continue;

	    printf("casper: raw disk thread rcvd data.\n");

        /* Read param struct for this block */
        ptr = guppi_databuf_header(db, curblock);
        if (first) {
            guppi_read_obs_params(ptr, &gp, &pf);
            first = 0;
        } else {
            guppi_read_subint_params(ptr, &gp, &pf);
        }

        /* Parse packet size, npacket from header */
        hgeti4(ptr, "PKTIDX", &packetidx);
        hgeti4(ptr, "PKTSIZE", &packetsize);
        hgeti4(ptr, "NPKT", &npacket);
        hgeti4(ptr, "NDROP", &ndrop);

        /* Check for re-quantization flag */
        int nbits_req = 0;
        if (hgeti4(ptr, "NBITSREQ", &nbits_req) == 0) {
            /* Param not present, don't requantize */
            requantize = 0;
        } else {
            /* Param is present */
            if (nbits_req==8)
                requantize = 0;
            else if (nbits_req==2) 
                requantize = 1;
            else
                /* Invalid selection for requested nbits 
                 * .. die or ignore?
                 */
                requantize = 0;
        }

        /* Set up data ptr for quant routines */
        pf.sub.data = (unsigned char *)guppi_databuf_data(db, curblock);

        /* Wait for packet 0 before starting write */
        if (got_packet_0==0 && packetidx==0 && gp.stt_valid==1) {
            got_packet_0 = 1;
            guppi_read_obs_params(ptr, &gp, &pf);
            orig_blocksize = pf.sub.bytes_per_subint;
            char fname[256];
            sprintf(fname, "%s.%4.4d.raw", pf.basefilename, filenum);
            fprintf(stderr, "Opening raw file '%s'\n", fname);
            // TODO: check for file exist.
            fraw = fopen(fname, "w");
            if (fraw==NULL) {
                guppi_error("guppi_rawdisk_thread", "Error opening file.");
                pthread_exit(NULL);
            }
            /* Determine scaling factors for quantization if appropriate */
            if (requantize) {
                mean = (double *)realloc(mean, 
                        pf.hdr.rcvr_polns * pf.hdr.nchan * sizeof(double));
                std  = (double *)realloc(std,  
                        pf.hdr.rcvr_polns * pf.hdr.nchan * sizeof(double));
                compute_stat(&pf, mean, std);
                fprintf(stderr, "Computed 2-bit stats\n");
            }
        }
        
        /* See if we need to open next file */
        if (block_count >= blocks_per_file) {
            fclose(fraw);
            filenum++;
            char fname[256];
            sprintf(fname, "%s.%4.4d.raw", pf.basefilename, filenum);
            fprintf(stderr, "Opening raw file '%s'\n", fname);
            fraw = fopen(fname, "w");
            if (fraw==NULL) {
                guppi_error("guppi_rawdisk_thread", "Error opening file.");
                pthread_exit(NULL);
            }
            block_count=0;
        }

        /* See how full databuf is */
        //total_status = guppi_databuf_total_status(db);

        /* Requantize from 8 bits to 2 bits if necessary.
         * See raw_quant.c for more usage examples.
         */
        if (requantize && got_packet_0) {
            pf.sub.bytes_per_subint = orig_blocksize;
            /* Does the quantization in-place */
            quantize_2bit(&pf, mean, std);
            /* Update some parameters for output */
            hputi4(ptr, "BLOCSIZE", pf.sub.bytes_per_subint);
            hputi4(ptr, "NBITS", pf.hdr.nbits);
        }

        /* Get full data block size */
        hgeti4(ptr, "BLOCSIZE", &blocksize);

        /* If we got packet 0, write data to disk */
        if (got_packet_0) { 

            /* Note waiting status */
            guppi_status_lock_safe(&st);
            hputs(st.buf, STATUS_KEY, "writing");
            guppi_status_unlock_safe(&st);

            /* Write header to file */
            hend = ksearch(ptr, "END");
            for (ptr=ptr; ptr<=hend; ptr+=80) {
                fwrite(ptr, 80, 1, fraw);
            }

            /* Write data */
            ptr = guppi_databuf_data(db, curblock);
            rv = fwrite(ptr, 1, (size_t)blocksize, fraw);
            if (rv != blocksize) { 
                guppi_error("guppi_rawdisk_thread", 
                        "Error writing data.");
            }

            /* Increment counter */
            block_count++;

            /* flush output */
            fflush(fraw);
        }

        /* Mark as free */
        guppi_databuf_set_free(db, curblock);

        /* Go to next block */
        curblock = (curblock + 1) % db->n_block;

        /* Check for cancel */
        pthread_testcancel();

    }

    pthread_exit(NULL);

    pthread_cleanup_pop(0); /* Closes fclose */
    pthread_cleanup_pop(0); /* Closes guppi_databuf_detach */
    pthread_cleanup_pop(0); /* Closes guppi_free_psrfits */
    pthread_cleanup_pop(0); /* Closes set_exit_status */
    pthread_cleanup_pop(0); /* Closes guppi_status_detach */

}
