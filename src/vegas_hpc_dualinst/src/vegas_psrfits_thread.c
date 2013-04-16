/* vegas_psrfits_thread.c
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
#include "polyco.h"
#include "psrfits.h"
#include "fold.h"
#include "vegas_error.h"
#include "vegas_status.h"
#include "vegas_databuf.h"

#define STATUS_KEY "DISKSTAT"
#include "vegas_threads.h"

// Read a status buffer all of the key observation paramters
extern void vegas_read_obs_params(char *buf, 
                                  struct vegas_params *g, 
                                  struct psrfits *p);

/* Parse info from buffer into param struct */
extern void vegas_read_subint_params(char *buf, 
                                     struct vegas_params *g,
                                     struct psrfits *p);

/* Downsampling functions */
extern void get_stokes_I(struct psrfits *pf);
extern void downsample_freq(struct psrfits *pf);
extern void downsample_time(struct psrfits *pf);
extern void vegas_update_ds_params(struct psrfits *pf);

void zero_end_chans(struct psrfits *pf)
{
    int ii, jj;
    struct hdrinfo *hdr = &(pf->hdr);
    char *data = (char *)pf->sub.data;
    const int nchan = hdr->nchan;
    const int nspec = hdr->nsblk * hdr->npol;
    
    for (ii = 0, jj = 0 ; ii < nspec ; ii++, jj += nchan)
        data[jj] = data[jj+nchan-1] = 0;
}


void vegas_psrfits_thread(void *_args) {
    
    /* Get args */
    struct vegas_thread_args *args = (struct vegas_thread_args *)_args;
    pthread_cleanup_push((void *)vegas_thread_set_finished, args);
    
    /* Set cpu affinity */
    cpu_set_t cpuset, cpuset_orig;
    sched_getaffinity(0, sizeof(cpu_set_t), &cpuset_orig);
    CPU_ZERO(&cpuset);
    CPU_SET(1, &cpuset);
    int rv = sched_setaffinity(0, sizeof(cpu_set_t), &cpuset);
    if (rv<0) { 
        vegas_error("vegas_psrfits_thread", "Error setting cpu affinity.");
        perror("sched_setaffinity");
    }

    /* Set priority */
    rv = setpriority(PRIO_PROCESS, 0, args->priority);
    if (rv<0) {
        vegas_error("vegas_psrfits_thread", "Error setting priority level.");
        perror("set_priority");
    }
    
    /* Attach to status shared mem area */
    struct vegas_status st;
    rv = vegas_status_attach(&st);
    if (rv!=VEGAS_OK) {
        vegas_error("vegas_psrfits_thread", 
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
    struct psrfits pf;
    pf.sub.data = NULL;
    pf.sub.dat_freqs = pf.sub.dat_weights = NULL;
    pf.sub.dat_offsets = pf.sub.dat_scales = NULL;
    pf.hdr.chan_dm = 0.0;
    pf.filenum = 0; // This is crucial
    pthread_cleanup_push((void *)vegas_free_psrfits, &pf);
    pthread_cleanup_push((void *)psrfits_close, &pf);
    //pf.multifile = 0;  // Use a single file for fold mode
    pf.multifile = 1;  // Use a multiple files for fold mode
    pf.quiet = 0;      // Print a message per each subint written
    
    /* Attach to databuf shared mem */
    struct vegas_databuf *db;
    db = vegas_databuf_attach(args->input_buffer);
    if (db==NULL) {
        vegas_error("vegas_psrfits_thread",
                    "Error attaching to databuf shared memory.");
        pthread_exit(NULL);
    }
    pthread_cleanup_push((void *)vegas_databuf_detach, db);
    
    /* Loop */
    int curblock=0, total_status=0, firsttime=1, run=1, got_packet_0=0;
    int mode=SEARCH_MODE;
    char *ptr;
    char tmpstr[256];
    struct foldbuf fb;
    struct polyco pc[64];  
    memset(pc, 0, sizeof(pc));
    int n_polyco_written=0;
    float *fold_output_array = NULL;
    int scan_finished=0;
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
        hputi4(st.buf, "CURBLOCK", curblock);
        vegas_status_unlock_safe(&st);

        /* See how full databuf is */
        total_status = vegas_databuf_total_status(db);
        
        /* Read param structs for this block */
        ptr = vegas_databuf_header(db, curblock);
        if (firsttime) {
            vegas_read_obs_params(ptr, &gp, &pf);
            firsttime = 0;
        } else {
            vegas_read_subint_params(ptr, &gp, &pf);
        }

        /* Find out what mode this data is in */
        mode = psrfits_obs_mode(pf.hdr.obs_mode);

        /* Check if we got both packet 0 and a valid observation
         * start time.  If so, flag writing to start.
         */
        if (got_packet_0==0 && gp.packetindex==0 && gp.stt_valid==1) {
            got_packet_0 = 1;
            vegas_read_obs_params(ptr, &gp, &pf);
            vegas_update_ds_params(&pf);
            memset(pc, 0, sizeof(pc));
            n_polyco_written=0;
        }

        /* If actual observation has started, write the data */
        if (got_packet_0) { 

            /* Note waiting status */
            vegas_status_lock_safe(&st);
            hputs(st.buf, STATUS_KEY, "writing");
            vegas_status_unlock_safe(&st);
            
            /* Get the pointer to the current data */
            if (mode==FOLD_MODE) {
                fb.nchan = pf.hdr.nchan;
                fb.npol = pf.hdr.npol;
                fb.nbin = pf.hdr.nbin;
                fb.order = pol_bin_chan; // XXX fix this!
                fb.data = (float *)vegas_databuf_data(db, curblock);
                fb.count = (unsigned *)(vegas_databuf_data(db, curblock)
                        + foldbuf_data_size(&fb));
                fold_output_array = (float *)realloc(fold_output_array,
                        sizeof(float) * pf.hdr.nbin * pf.hdr.nchan * 
                        pf.hdr.npol);
                pf.sub.data = (unsigned char *)fold_output_array;
                pf.fold.pc = (struct polyco *)(vegas_databuf_data(db,curblock)
                        + foldbuf_data_size(&fb) + foldbuf_count_size(&fb));
            } else 
                pf.sub.data = (unsigned char *)vegas_databuf_data(db, curblock);
            
            /* Set the DC and Nyquist channels explicitly to zero */
            /* because of the "FFT Problem" that splits DC power  */
            /* into those two bins.                               */
            // XXX why are we doing this? we should just set weight to 0
            //zero_end_chans(&pf);
#if 0
            /* Output only Stokes I (in place) */
            if (pf.hdr.onlyI && pf.hdr.npol==4)
                get_stokes_I(&pf);

            /* Downsample in frequency (in place) */
            if (pf.hdr.ds_freq_fact > 1)
                downsample_freq(&pf);

            /* Downsample in time (in place) */
            if (pf.hdr.ds_time_fact > 1)
                downsample_time(&pf);

            /* Folded data needs a transpose */
            if (mode==FOLD_MODE)
                normalize_transpose_folds(fold_output_array, &fb);
#endif
            /* Write the data */
            int last_filenum = pf.filenum;
            psrfits_write_subint(&pf);

            /* Any actions that need to be taken when a new file
             * is created.
             */
            if (pf.filenum!=last_filenum) {
                /* No polycos yet written to the new file */
                n_polyco_written=0;
            }

            /* Write the polycos if needed */
            int write_pc=0, i, j;
            for (i=0; i<pf.fold.n_polyco_sets; i++) {
                if (pf.fold.pc[i].used==0) continue; 
                int new_pc=1;
                for (j=0; j<n_polyco_written; j++) {
                    if (polycos_differ(&pf.fold.pc[i], &pc[j])==0) {
                        new_pc=0;
                        break;
                    }
                }
                if (new_pc || n_polyco_written==0) {
                    pc[n_polyco_written] = pf.fold.pc[i];
                    n_polyco_written++;
                    write_pc=1;
                } else {
                    pf.fold.pc[i].used = 0; // Already have this one
                }
            }
 //           if (write_pc) 
 //               psrfits_write_polycos(&pf, pf.fold.pc, pf.fold.n_polyco_sets);

            /* Is the scan complete? */
            if ((pf.hdr.scanlen > 0.0) && 
                (pf.T > pf.hdr.scanlen)) scan_finished = 1;
            
            /* For debugging... */
            if (gp.drop_frac > 0.0) {
               printf("Block %d dropped %.3g%% of the packets\n", 
                      pf.tot_rows, gp.drop_frac*100.0);
            }

        }

        /* Mark as free */
        vegas_databuf_set_free(db, curblock);
        
        /* Go to next block */
        curblock = (curblock + 1) % db->n_block;
        
        /* Check for cancel */
        pthread_testcancel();
        
    } while (run && !scan_finished);
    
    /* Cleanup */
    
    if (fold_output_array!=NULL) free(fold_output_array);

    pthread_exit(NULL);
    
    pthread_cleanup_pop(0); /* Closes psrfits_close */
    pthread_cleanup_pop(0); /* Closes vegas_free_psrfits */
    pthread_cleanup_pop(0); /* Closes set_exit_status */
    pthread_cleanup_pop(0); /* Closes set_finished */
    pthread_cleanup_pop(0); /* Closes vegas_status_detach */
    pthread_cleanup_pop(0); /* Closes vegas_databuf_detach */
}
