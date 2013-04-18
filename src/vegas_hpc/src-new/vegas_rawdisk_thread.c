/* vegas_rawdisk_thread.c
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
#include "vegas_error.h"
#include "vegas_status.h"
#include "vegas_databuf.h"

#define STATUS_KEY "DISKSTAT"
#include "vegas_threads.h"
#include "vegas_defines.h"

#if FITS_TYPE != SDFITS
    #error "FITS_TYPE must be set to SDFITS"
#endif

#include "sdfits.h"

// Read a status buffer all of the key observation paramters
extern void vegas_read_obs_params(char *buf, 
                                  struct vegas_params *g, 
                                  struct sdfits *p);

/* Parse info from buffer into param struct */
extern void vegas_read_subint_params(char *buf, 
                                     struct vegas_params *g,
                                     struct sdfits *p);

int safe_fclose(FILE *f) {
    if (f==NULL) return 0;
    sync();
    return fclose(f);
}

void vegas_rawdisk_thread(void *_args) {

    /* Set cpu affinity */
    cpu_set_t cpuset, cpuset_orig;
    sched_getaffinity(0, sizeof(cpu_set_t), &cpuset_orig);
    CPU_ZERO(&cpuset);
    CPU_SET(6, &cpuset);
    int rv = sched_setaffinity(0, sizeof(cpu_set_t), &cpuset);
    if (rv<0) { 
        vegas_error("vegas_rawdisk_thread", "Error setting cpu affinity.");
        perror("sched_setaffinity");
    }

    /* Get args */
    struct vegas_thread_args *args = (struct vegas_thread_args *)_args;

    /* Set priority */
    rv = setpriority(PRIO_PROCESS, 0, 0);
    if (rv<0) {
        vegas_error("vegas_rawdisk_thread", "Error setting priority level.");
        perror("set_priority");
    }

    /* Attach to status shared mem area */
    struct vegas_status st;
    rv = vegas_status_attach(&st);
    if (rv!=VEGAS_OK) {
        vegas_error("vegas_rawdisk_thread", 
                "Error attaching to status shared memory.");
        pthread_exit(NULL);
    }
    pthread_cleanup_push((void *)vegas_status_detach, &st);
    pthread_cleanup_push((void *)set_exit_status, &st);

    /* Init status */
    vegas_status_lock_safe(&st);
    hputs(st.buf, STATUS_KEY, "init");
    vegas_status_unlock_safe(&st);

    /* Read in general parameters */
    struct vegas_params gp;
    struct sdfits sf;
    pthread_cleanup_push((void *)vegas_free_sdfits, &sf);

    /* Attach to databuf shared mem */
    struct vegas_databuf *db;
    db = vegas_databuf_attach(args->input_buffer);
    if (db==NULL) {
        vegas_error("vegas_rawdisk_thread",
                "Error attaching to databuf shared memory.");
        pthread_exit(NULL);
    }
    pthread_cleanup_push((void *)vegas_databuf_detach, db);

    /* Init output file */
    FILE *fraw = NULL;
    pthread_cleanup_push((void *)safe_fclose, fraw);

    /* Loop */
    int blocksize=0;
    int curblock=0, dataset;
    int block_count=0, blocks_per_file=128, filenum=0;
    int first=1;
    char *ptr;
    float *data_array;
    struct databuf_index* db_index;

    signal(SIGINT,cc);

    while (run) {

        /* Note waiting status */
        vegas_status_lock_safe(&st);
        hputs(st.buf, STATUS_KEY, "waiting");
        vegas_status_unlock_safe(&st);

        /* Wait for buf to have data */
        rv = vegas_databuf_wait_filled(db, curblock);
        if (rv!=0) continue;

        /* Read param struct and index for this block */
        ptr = vegas_databuf_header(db, curblock);
        db_index = (struct databuf_index*)(vegas_databuf_index(db, curblock));

        /* If first time running */
        if (first==1)
        {
            first = 0;
            vegas_read_obs_params(ptr, &gp, &sf);

            char fname[256];
            sprintf(fname, "%s_%4d.raw", sf.basefilename, filenum);
            fprintf(stderr, "Opening raw file '%s'\n", fname);
            // TODO: check for file exist.
            fraw = fopen(fname, "wb");
            if (fraw==NULL) {
                vegas_error("vegas_rawdisk_thread", "Error opening file.");
                pthread_exit(NULL);
            }
        }
        else
            vegas_read_subint_params(ptr, &gp, &sf);

        /* See if we need to open next file */
        if (block_count >= blocks_per_file) {
            fclose(fraw);
            filenum++;
            char fname[256];
            sprintf(fname, "%s_%4d.raw", sf.basefilename, filenum);
            fprintf(stderr, "Opening raw file '%s'\n", fname);
            fraw = fopen(fname, "wb");
            if (fraw==NULL) {
                vegas_error("vegas_rawdisk_thread", "Error opening file.");
                pthread_exit(NULL);
            }
            block_count=0;
        }

        /* Get full data block size */
        hgeti4(ptr, "BLOCSIZE", &blocksize);

        /* Note writing status and current block */
        vegas_status_lock_safe(&st);
        hputi4(st.buf, "DSKBLKIN", curblock);
        hputs(st.buf, STATUS_KEY, "writing");
        vegas_status_unlock_safe(&st);

        /* Write all data arrays to disk */
        for(dataset = 0; dataset < db_index->num_datasets; dataset++)
        {
            data_array = (float*)(vegas_databuf_data(db, curblock) +
                                     db_index->disk_buf[dataset].array_offset);

            rv = fwrite(data_array, 4, (size_t)(db_index->array_size/4), fraw);

            if (rv != db_index->array_size/4) { 
                vegas_error("vegas_rawdisk_thread", 
                        "Error writing data.");
            }
        }

        /* Increment block counter */
        block_count++;

        /* flush output */
        fflush(fraw);

        /* Mark as free */
        vegas_databuf_set_free(db, curblock);

        /* Go to next block */
        curblock = (curblock + 1) % db->n_block;

        /* Check for cancel */
        pthread_testcancel();

    }

    pthread_exit(NULL);

    pthread_cleanup_pop(0); /* Closes fclose */
    pthread_cleanup_pop(0); /* Closes vegas_databuf_detach */
    pthread_cleanup_pop(0); /* Closes vegas_free_psrfits */
    pthread_cleanup_pop(0); /* Closes set_exit_status */
    pthread_cleanup_pop(0); /* Closes vegas_status_detach */

}
