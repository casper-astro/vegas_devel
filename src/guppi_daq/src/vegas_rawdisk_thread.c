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
#include "guppi_error.h"
#include "guppi_status.h"
#include "guppi_databuf.h"

#define STATUS_KEY "DISKSTAT"
#include "guppi_threads.h"
#include "guppi_defines.h"

#if FITS_TYPE != SDFITS
    #error "FITS_TYPE must be set to SDFITS"
#endif

#include "sdfits.h"

// Read a status buffer all of the key observation paramters
extern void guppi_read_obs_params(char *buf, 
                                  struct guppi_params *g, 
                                  struct sdfits *p);

/* Parse info from buffer into param struct */
extern void guppi_read_subint_params(char *buf, 
                                     struct guppi_params *g,
                                     struct sdfits *p);

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
    struct sdfits sf;
    pthread_cleanup_push((void *)guppi_free_sdfits, &sf);

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
        guppi_status_lock_safe(&st);
        hputs(st.buf, STATUS_KEY, "waiting");
        guppi_status_unlock_safe(&st);

        /* Wait for buf to have data */
        rv = guppi_databuf_wait_filled(db, curblock);
        if (rv!=0) continue;

        /* Read param struct and index for this block */
        ptr = guppi_databuf_header(db, curblock);
        db_index = (struct databuf_index*)(guppi_databuf_index(db, curblock));

        /* If first time running */
        if (first==1)
        {
            first = 0;
            guppi_read_obs_params(ptr, &gp, &sf);

            char fname[256];
            sprintf(fname, "%s_%4d.raw", sf.basefilename, filenum);
            fprintf(stderr, "Opening raw file '%s'\n", fname);
            // TODO: check for file exist.
            fraw = fopen(fname, "wb");
            if (fraw==NULL) {
                guppi_error("guppi_rawdisk_thread", "Error opening file.");
                pthread_exit(NULL);
            }
        }
        else
            guppi_read_subint_params(ptr, &gp, &sf);

        /* See if we need to open next file */
        if (block_count >= blocks_per_file) {
            fclose(fraw);
            filenum++;
            char fname[256];
            sprintf(fname, "%s_%4d.raw", sf.basefilename, filenum);
            fprintf(stderr, "Opening raw file '%s'\n", fname);
            fraw = fopen(fname, "wb");
            if (fraw==NULL) {
                guppi_error("guppi_rawdisk_thread", "Error opening file.");
                pthread_exit(NULL);
            }
            block_count=0;
        }

        /* Get full data block size */
        hgeti4(ptr, "BLOCSIZE", &blocksize);

        /* Note writing status */
        guppi_status_lock_safe(&st);
        hputs(st.buf, STATUS_KEY, "writing");
        guppi_status_unlock_safe(&st);

        /* Write all data arrays to disk */
        for(dataset = 0; dataset < db_index->num_datasets; dataset++)
        {
            data_array = (float*)(guppi_databuf_data(db, curblock) +
                                     db_index->disk_buf[dataset].array_offset);

            rv = fwrite(data_array, 4, (size_t)(db_index->array_size/4), fraw);

            if (rv != db_index->array_size/4) { 
                guppi_error("guppi_rawdisk_thread", 
                        "Error writing data.");
            }
        }

        /* Increment block counter */
        block_count++;

        /* flush output */
        fflush(fraw);

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
