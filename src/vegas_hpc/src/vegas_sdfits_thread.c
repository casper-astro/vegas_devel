/* vegas_sdfits_thread.c
 *
 * Write databuf blocks out to disk, in SDFITS format.
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
#include "sdfits.h"
#include "vegas_error.h"
#include "vegas_status.h"
#include "vegas_databuf.h"
#include "spead_heap.h"

#define STATUS_KEY "DISKSTAT"
#include "vegas_threads.h"

// Read a status buffer all of the key observation paramters
extern void vegas_read_obs_params(char *buf, 
                                  struct vegas_params *g, 
                                  struct sdfits *sf);

/* Parse info from buffer into param struct */
extern void vegas_read_subint_params(char *buf, 
                                     struct vegas_params *g,
                                     struct sdfits *sf);


void vegas_sdfits_thread(void *_args) {
    
    /* Get args */
    struct vegas_thread_args *args = (struct vegas_thread_args *)_args;
    pthread_cleanup_push((void *)vegas_thread_set_finished, args);
    
    /* Set cpu affinity */
    cpu_set_t cpuset, cpuset_orig;
    sched_getaffinity(0, sizeof(cpu_set_t), &cpuset_orig);
    CPU_ZERO(&cpuset);
    CPU_SET(6, &cpuset);
    int rv = sched_setaffinity(0, sizeof(cpu_set_t), &cpuset);
    if (rv<0) { 
        vegas_error("vegas_sdfits_thread", "Error setting cpu affinity.");
        perror("sched_setaffinity");
    }

    /* Set priority */
    rv = setpriority(PRIO_PROCESS, 0, args->priority);
    if (rv<0) {
        vegas_error("vegas_sdfits_thread", "Error setting priority level.");
        perror("set_priority");
    }
    
    /* Attach to status shared mem area */
    struct vegas_status st;
    rv = vegas_status_attach(&st);
    if (rv!=VEGAS_OK) {
        vegas_error("vegas_sdfits_thread", 
                    "Error attaching to status shared memory.");
        pthread_exit(NULL);
    }
    pthread_cleanup_push((void *)vegas_status_detach, &st);
    pthread_cleanup_push((void *)set_exit_status, &st);
    
    /* Init status */
    vegas_status_lock_safe(&st);
    hputs(st.buf, STATUS_KEY, "init");
    vegas_status_unlock_safe(&st);
    
    /* Initialize some key parameters */
    struct vegas_params gp;
    struct sdfits sf;
    sf.data_columns.data = NULL;
    sf.filenum = 0;
    sf.new_file = 1; // This is crucial
    pthread_cleanup_push((void *)vegas_free_sdfits, &sf);
    pthread_cleanup_push((void *)sdfits_close, &sf);
    //pf.multifile = 0;  // Use a single file for fold mode
    sf.multifile = 1;  // Use a multiple files for fold mode
    sf.quiet = 0;      // Print a message per each subint written
    
    /* Attach to databuf shared mem */
    struct vegas_databuf *db;
    db = vegas_databuf_attach(args->input_buffer);
    if (db==NULL) {
        vegas_error("vegas_sdfits_thread",
                    "Error attaching to databuf shared memory.");
        pthread_exit(NULL);
    }
    pthread_cleanup_push((void *)vegas_databuf_detach, db);
    
    /* Loop */
    int curblock=0, total_status=0, firsttime=1, run=1, got_packet_0=0, dataset=0;
    char *ptr;
    char tmpstr[256];
    int scan_finished=0, first_heap_in_blk, old_filenum;
    int num_exposures_written = 0;
    int old_integ_num = -1;

    signal(SIGINT, cc);
    do {
        /* Note waiting status */
        vegas_status_lock_safe(&st);
        if (got_packet_0)
            sprintf(tmpstr, "waiting(%d)", curblock);
        else
            sprintf(tmpstr, "ready");
        hputs(st.buf, STATUS_KEY, tmpstr);
        vegas_status_unlock_safe(&st);
        
        /* Wait for buf to have data */
        rv = vegas_databuf_wait_filled(db, curblock);
        if (rv!=0) {
            // This is a big ol' kludge to avoid this process hanging
            // due to thread synchronization problems.
            sleep(1);
            continue; 
        }

        /* Note current block */
        vegas_status_lock_safe(&st);
        hputi4(st.buf, "DSKBLKIN", curblock);
        vegas_status_unlock_safe(&st);

        /* See how full databuf is */
        total_status = vegas_databuf_total_status(db);
        
        /* Read param structs for this block */
        ptr = vegas_databuf_header(db, curblock);
        if (firsttime) {
            vegas_read_obs_params(ptr, &gp, &sf);
            firsttime = 0;
        } else {
            vegas_read_subint_params(ptr, &gp, &sf);
        }

        /* Note waiting status */
        vegas_status_lock_safe(&st);
        hputs(st.buf, STATUS_KEY, "writing");
        vegas_status_unlock_safe(&st);

        struct sdfits_data_columns* data_cols;
        struct databuf_index* db_index;

        db_index = (struct databuf_index*)(vegas_databuf_index(db, curblock));

        /* Read the block index, writing each dataset to a SDFITS file */
        for(dataset = 0; dataset < db_index->num_datasets; dataset++)
        {
            data_cols = (struct sdfits_data_columns*)(vegas_databuf_data(db, curblock) +
                        db_index->disk_buf[dataset].struct_offset);

            sf.data_columns = *data_cols;

            /* Write the data */
            old_filenum = sf.filenum;
            sdfits_write_subint(&sf);

            /*Write new file number to shared memory*/
            if(sf.filenum != old_filenum)
                hputi4(st.buf, "FILENUM", sf.filenum);

            /* If a new integration number, increment the number of exposures written */
            if(data_cols->integ_num != old_integ_num)
            {
                num_exposures_written += 1;
                old_integ_num = data_cols->integ_num;
            }

        }

        /* Indicate number of exposures written */
        vegas_status_lock_safe(&st);
        hputi4(st.buf, "DSKEXPWR", num_exposures_written);
        vegas_status_unlock_safe(&st);

        /* For debugging... */
        if (gp.drop_frac > 0.0) {
            printf("Block %d dropped %.3g%% of the packets\n", 
                    sf.tot_rows, gp.drop_frac*100.0);
        }

        /* Mark as free */
        vegas_databuf_set_free(db, curblock);
        
        /* Go to next block */
        curblock = (curblock + 1) % db->n_block;
        
        /* Check for cancel */
        pthread_testcancel();
        
    } while (run && !scan_finished);
    
    /* Cleanup */
    pthread_exit(NULL);
    
    pthread_cleanup_pop(0); /* Closes sdfits_close */
    pthread_cleanup_pop(0); /* Closes vegas_free_sdfits */
    pthread_cleanup_pop(0); /* Closes set_exit_status */
    pthread_cleanup_pop(0); /* Closes set_finished */
    pthread_cleanup_pop(0); /* Closes vegas_status_detach */
    pthread_cleanup_pop(0); /* Closes vegas_databuf_detach */
}
