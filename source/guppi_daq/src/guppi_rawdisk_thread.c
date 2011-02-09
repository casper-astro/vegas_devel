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

    /* Loop */
    int packetidx=0, npacket=0, ndrop=0, packetsize=0, blocksize=0;
    int curblock=0;
    int block_count=0, blocks_per_file=128, filenum=0;
    int got_packet_0=0, first=1;
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
        hgeti4(ptr, "BLOCSIZE", &blocksize);

        /* Wait for packet 0 before starting write */
        if (got_packet_0==0 && packetidx==0 && gp.stt_valid==1) {
            got_packet_0 = 1;
            guppi_read_obs_params(ptr, &gp, &pf);
            char fname[256];
            sprintf(fname, "%s.%4.4d.raw", pf.basefilename, filenum);
            fprintf(stderr, "Opening raw file '%s'\n", fname);
            // TODO: check for file exist.
            fraw = fopen(fname, "w");
            if (fraw==NULL) {
                guppi_error("guppi_rawdisk_thread", "Error opening file.");
                pthread_exit(NULL);
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
