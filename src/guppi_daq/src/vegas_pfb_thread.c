/* vegas_pfb_thread.c
 *
 * Performs PFB on incoming time samples
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

#include "fitshead.h"
#include "sdfits.h"
#include "guppi_error.h"
#include "guppi_status.h"
#include "guppi_databuf.h"
#include "guppi_params.h"
#include "pfb_gpu.h"

#define STATUS_KEY "GPUSTAT"
#include "guppi_threads.h"

/* Parse info from buffer into param struct */
extern void guppi_read_subint_params(char *buf, 
                                     struct guppi_params *g,
                                     struct sdfits *p);
extern void guppi_read_obs_params(char *buf, 
                                     struct guppi_params *g,
                                     struct sdfits *p);

void vegas_pfb_thread(void *_args) {

    /* Get args */
    struct guppi_thread_args *args = (struct guppi_thread_args *)_args;
    int rv;

    /* Set cpu affinity */
    cpu_set_t cpuset, cpuset_orig;
    sched_getaffinity(0, sizeof(cpu_set_t), &cpuset_orig);
    //CPU_ZERO(&cpuset);
    CPU_CLR(13, &cpuset);
    CPU_SET(11, &cpuset);
    rv = sched_setaffinity(0, sizeof(cpu_set_t), &cpuset);
    if (rv<0) { 
        guppi_error("vegas_pfb_thread", "Error setting cpu affinity.");
        perror("sched_setaffinity");
    }

    /* Set priority */
    rv = setpriority(PRIO_PROCESS, 0, args->priority);
    if (rv<0) {
        guppi_error("vegas_pfb_thread", "Error setting priority level.");
        perror("set_priority");
    }

    /* Attach to status shared mem area */
    struct guppi_status st;
    rv = guppi_status_attach(&st);
    if (rv!=GUPPI_OK) {
        guppi_error("vegas_pfb_thread", 
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

    /* Init structs */
    struct guppi_params gp;
    struct sdfits sf;
    pthread_cleanup_push((void *)guppi_free_sdfits, &sf);

    /* Attach to databuf shared mem */
    struct guppi_databuf *db_in, *db_out;
    db_in = guppi_databuf_attach(args->input_buffer);
    if (db_in==NULL) {
        char msg[256];
        sprintf(msg, "Error attaching to databuf(%d) shared memory.",
                args->input_buffer);
        guppi_error("vegas_pfb_thread", msg);
        pthread_exit(NULL);
    }
    pthread_cleanup_push((void *)guppi_databuf_detach, db_in);
    db_out = guppi_databuf_attach(args->output_buffer);
    if (db_out==NULL) {
        char msg[256];
        sprintf(msg, "Error attaching to databuf(%d) shared memory.",
                args->output_buffer);
        guppi_error("vegas_pfb_thread", msg);
        pthread_exit(NULL);
    }
    pthread_cleanup_push((void *)guppi_databuf_detach, db_out);

    /* Loop */
    char *hdr_in = NULL;
    int curblock_in=0;
    int first=1;
    int acc_len = 0;
    int nchan = 0;
    int nsubband = 0;
    signal(SIGINT,cc);

    guppi_status_lock_safe(&st);
    if (hgeti4(st.buf, "NCHAN", &nchan)==0) {
        fprintf(stderr, "ERROR: %s not in status shm!\n", "NCHAN");
    }
    if (hgeti4(st.buf, "NSUBBAND", &nsubband)==0) {
        fprintf(stderr, "ERROR: %s not in status shm!\n", "NSUBBAND");
    }
    guppi_status_unlock_safe(&st);
    if (EXIT_SUCCESS != init_gpu(db_in->block_size,
                                 db_out->block_size,
                                 nsubband,
                                 nchan))
    {
        (void) fprintf(stderr, "ERROR: GPU initialisation failed!\n");
        run = 0;
    }

    while (run) {

        /* Note waiting status */
        guppi_status_lock_safe(&st);
        hputs(st.buf, STATUS_KEY, "waiting");
        guppi_status_unlock_safe(&st);

        /* Wait for buf to have data */
        rv = guppi_databuf_wait_filled(db_in, curblock_in);
        if (rv!=0) continue;

        /* Note waiting status, current input block */
        guppi_status_lock_safe(&st);
        hputs(st.buf, STATUS_KEY, "processing");
        hputi4(st.buf, "PFBBLKIN", curblock_in);
        guppi_status_unlock_safe(&st);

        hdr_in = guppi_databuf_header(db_in, curblock_in);
        
        /* Get params */
        if (first)
        {
            guppi_read_obs_params(hdr_in, &gp, &sf);
            /* Read required exposure from status shared memory, and calculate
               corresponding accumulation length */
            acc_len = (abs(sf.hdr.chan_bw) * sf.hdr.hwexposr);
        }
        guppi_read_subint_params(hdr_in, &gp, &sf);

        /* Call PFB function */
        do_pfb(db_in, curblock_in, db_out, first, st, acc_len);

        /* Mark input block as free */
        guppi_databuf_set_free(db_in, curblock_in);
        /* Go to next input block */
        curblock_in = (curblock_in + 1) % db_in->n_block;

        /* Check for cancel */
        pthread_testcancel();

        if (first) {
            first=0;
        }
    }
    run=0;

    //cudaThreadExit();
    pthread_exit(NULL);

    cleanup_gpu();

    pthread_cleanup_pop(0); /* Closes guppi_databuf_detach(out) */
    pthread_cleanup_pop(0); /* Closes guppi_databuf_detach(in) */
    pthread_cleanup_pop(0); /* Closes guppi_free_sdfits */
    pthread_cleanup_pop(0); /* Closes guppi_thread_set_finished */
    pthread_cleanup_pop(0); /* Closes set_exit_status */
    pthread_cleanup_pop(0); /* Closes guppi_status_detach */

}

