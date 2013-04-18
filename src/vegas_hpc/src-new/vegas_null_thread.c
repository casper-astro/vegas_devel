/* vegas_null_thread.c
 *
 * Marks databufs empty as soon as they're full
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
#include "vegas_error.h"
#include "vegas_status.h"
#include "vegas_databuf.h"
#include "vegas_params.h"

#define STATUS_KEY "NULLSTAT"
#include "vegas_threads.h"

#if FITS_TYPE == PSRFITS
#include "psrfits.h"
#else
#include "sdfits.h"
#endif


#if FITS_TYPE == PSRFITS
// Read a status buffer all of the key observation paramters
extern void vegas_read_obs_params(char *buf, 
                                  struct vegas_params *g, 
                                  struct psrfits *p);

/* Parse info from buffer into param struct */
extern void vegas_read_subint_params(char *buf, 
                                     struct vegas_params *g,
                                     struct psrfits *p);
#else
// Read a status buffer all of the key observation paramters
extern void vegas_read_obs_params(char *buf, 
                                  struct vegas_params *g, 
                                  struct sdfits *p);

/* Parse info from buffer into param struct */
extern void vegas_read_subint_params(char *buf, 
                                     struct vegas_params *g,
                                     struct sdfits *p);
#endif


void vegas_null_thread(void *_args) {

    int rv;
    /* Set cpu affinity */
    cpu_set_t cpuset, cpuset_orig;
    sched_getaffinity(0, sizeof(cpu_set_t), &cpuset_orig);
    CPU_ZERO(&cpuset);
    CPU_SET(6, &cpuset);
    rv = sched_setaffinity(0, sizeof(cpu_set_t), &cpuset);
    if (rv<0) { 
        vegas_error("vegas_null_thread", "Error setting cpu affinity.");
        perror("sched_setaffinity");
    }

    /* Set priority */
    rv = setpriority(PRIO_PROCESS, 0, 0);
    if (rv<0) {
        vegas_error("vegas_null_thread", "Error setting priority level.");
        perror("set_priority");
    }

    /* Get args */
    struct vegas_thread_args *args = (struct vegas_thread_args *)_args;

    /* Attach to status shared mem area */
    struct vegas_status st;
    rv = vegas_status_attach(&st);
    if (rv!=VEGAS_OK) {
        vegas_error("vegas_null_thread", 
                "Error attaching to status shared memory.");
        pthread_exit(NULL);
    }
    pthread_cleanup_push((void *)vegas_status_detach, &st);
    pthread_cleanup_push((void *)set_exit_status, &st);

    /* Init status */
    vegas_status_lock_safe(&st);
    hputs(st.buf, STATUS_KEY, "init");
    vegas_status_unlock_safe(&st);

    /* Attach to databuf shared mem */
    struct vegas_databuf *db;
    db = vegas_databuf_attach(args->input_buffer);
    if (db==NULL) {
        vegas_error("vegas_null_thread",
                "Error attaching to databuf shared memory.");
        pthread_exit(NULL);
    }
    pthread_cleanup_push((void *)vegas_databuf_detach, db);

    /* Loop */
    char *ptr;
    struct vegas_params gp;
#if FITS_TYPE == PSRFITS
    struct psrfits pf;
    pf.sub.dat_freqs = NULL;
    pf.sub.dat_weights = NULL;
    pf.sub.dat_offsets = NULL;
    pf.sub.dat_scales = NULL;
    pthread_cleanup_push((void *)vegas_free_psrfits, &pf);
#else
    struct sdfits pf;
    pthread_cleanup_push((void *)vegas_free_sdfits, &pf);
#endif
    int curblock=0;
    signal(SIGINT,cc);
    while (run) {

        /* Note waiting status */
        vegas_status_lock_safe(&st);
        hputs(st.buf, STATUS_KEY, "waiting");
        vegas_status_unlock_safe(&st);

        /* Wait for buf to have data */
        rv = vegas_databuf_wait_filled(db, curblock);
        if (rv!=0) {
            //sleep(1);
            continue;
        }

        /* Note waiting status, current block */
        vegas_status_lock_safe(&st);
        hputs(st.buf, STATUS_KEY, "discarding");
        hputi4(st.buf, "DSKBLKIN", curblock);
        vegas_status_unlock_safe(&st);

        /* Get params */
        ptr = vegas_databuf_header(db, curblock);
        vegas_read_obs_params(ptr, &gp, &pf);

        /* Output if data was lost */
#if FITS_TYPE == PSRFITS
        if (gp.n_dropped!=0 && 
                (gp.packetindex==0 || strcmp(pf.hdr.obs_mode,"SEARCH"))) {
            printf("Block beginning with pktidx=%lld dropped %d packets\n",
                    gp.packetindex, gp.n_dropped);
            fflush(stdout);
        }
#else
        if (gp.num_pkts_dropped!=0 && gp.num_pkts_rcvd!=0) {
            printf("Block received %d packets and dropped %d packets\n",
                    gp.num_pkts_rcvd, gp.num_pkts_dropped);
            fflush(stdout);
        }
#endif

        /* Mark as free */
        vegas_databuf_set_free(db, curblock);

        /* Go to next block */
        curblock = (curblock + 1) % db->n_block;

        /* Check for cancel */
        pthread_testcancel();

    }

    pthread_exit(NULL);

    pthread_cleanup_pop(0); /* Closes set_exit_status */
    pthread_cleanup_pop(0); /* Closes vegas_free_psrfits */
    pthread_cleanup_pop(0); /* Closes vegas_status_detach */
    pthread_cleanup_pop(0); /* Closes vegas_databuf_detach */

}
