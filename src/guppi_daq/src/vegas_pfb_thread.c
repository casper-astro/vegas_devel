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
    char *hdr_in=NULL, *hdr_out=NULL;
    char *curdata_in, *curdata_out;
    struct databuf_index *curindex_in, *curindex_out;
    int curblock_in=0, curblock_out=0;
    int first=1;
    signal(SIGINT,cc);
    while (run) {

        /* Note waiting status */
        guppi_status_lock_safe(&st);
        hputs(st.buf, STATUS_KEY, "waiting");
        guppi_status_unlock_safe(&st);

        /* Wait for buf to have data */
        rv = guppi_databuf_wait_filled(db_in, curblock_in);
        if (rv!=0) continue;

        /* Note waiting status, current block */
        guppi_status_lock_safe(&st);
        hputs(st.buf, STATUS_KEY, "processing");
        hputi4(st.buf, "CURBLOCK", curblock_in);
        guppi_status_unlock_safe(&st);

        /* Get params */
        hdr_in = guppi_databuf_header(db_in, curblock_in);
        if (first)
            guppi_read_obs_params(hdr_in, &gp, &sf);
        else 
            guppi_read_subint_params(hdr_in, &gp, &sf);

        /* Any first-time init stuff */
        if (first) {

            /* Init PFB on GPU */
            init_pfb(db_in->block_size, db_out->block_size, db_in->index_size, sf.hdr.nchan);

            /* Clear first time flag */
            first=0;
        }

        /* Setup input and output data block stuff */
        hdr_out = guppi_databuf_header(db_out, curblock_out);
        curdata_out = (char *)guppi_databuf_data(db_out, curblock_out);
        curindex_out = (struct databuf_index*)guppi_databuf_index(db_out, curblock_out);

        curdata_in = (char *)guppi_databuf_data(db_in, curblock_in);
        curindex_in = (struct databuf_index*)guppi_databuf_index(db_in, curblock_in);

        memcpy(hdr_out, guppi_databuf_header(db_in, curblock_in),
                GUPPI_STATUS_SIZE);

        /* Call PFB function */
        do_pfb(curdata_in, curdata_out, curindex_in, curindex_out);

        /* Mark blocks as free/filled */
        guppi_databuf_set_free(db_in, curblock_in);
        guppi_databuf_set_filled(db_out, curblock_out);

        /* Go to next input block */
        curblock_in = (curblock_in + 1) % db_in->n_block;

        printf("Debug: vegas_pfb_thread going to next output block\n");

        /*  Wait for next output block */
        curblock_out = (curblock_out + 1) % db_out->n_block;
        while ((rv=guppi_databuf_wait_free(db_out, curblock_out)!=0) && run) {
            guppi_status_lock_safe(&st);
            hputs(st.buf, STATUS_KEY, "blocked");
            guppi_status_unlock_safe(&st);
        }

        /* Check for cancel */
        pthread_testcancel();

    }
    run=0;

    //cudaThreadExit();
    pthread_exit(NULL);

    destroy_pfb();

    pthread_cleanup_pop(0); /* Closes guppi_databuf_detach(out) */
    pthread_cleanup_pop(0); /* Closes guppi_databuf_detach(in) */
    pthread_cleanup_pop(0); /* Closes guppi_free_sdfits */
    pthread_cleanup_pop(0); /* Closes guppi_thread_set_finished */
    pthread_cleanup_pop(0); /* Closes set_exit_status */
    pthread_cleanup_pop(0); /* Closes guppi_status_detach */

}
