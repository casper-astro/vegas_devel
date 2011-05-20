/* guppi_dedisp_thread.c
 *
 * Dedisperse incoming baseband data
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

#include <math.h>
#include <complex.h>
#include <fftw3.h>

#include "fitshead.h"
#include "psrfits.h"
#include "guppi_error.h"
#include "guppi_status.h"
#include "guppi_databuf.h"
#include "guppi_params.h"

#include <cuda.h>
#include <cuda_runtime.h>
#include "dedisperse_gpu.h"
#include "dedisperse_utils.h"
#include "downsample_gpu.h"

#define STATUS_KEY "DISPSTAT"
#include "guppi_threads.h"

/* Parse info from buffer into param struct */
extern void guppi_read_subint_params(char *buf, 
                                     struct guppi_params *g,
                                     struct psrfits *p);
extern void guppi_read_obs_params(char *buf, 
                                     struct guppi_params *g,
                                     struct psrfits *p);

void guppi_dedisp_ds_thread(void *_args) {

    /* Get args */
    struct guppi_thread_args *args = (struct guppi_thread_args *)_args;

    int rv;
#if 0 
    /* Set cpu affinity */
    cpu_set_t cpuset, cpuset_orig;
    sched_getaffinity(0, sizeof(cpu_set_t), &cpuset_orig);
    CPU_ZERO(&cpuset);
    CPU_SET(1, &cpuset);
    rv = sched_setaffinity(0, sizeof(cpu_set_t), &cpuset);
    if (rv<0) { 
        guppi_error("guppi_dedisp_thread", "Error setting cpu affinity.");
        perror("sched_setaffinity");
    }
#endif

    /* Set priority */
    rv = setpriority(PRIO_PROCESS, 0, args->priority);
    if (rv<0) {
        guppi_error("guppi_dedisp_thread", "Error setting priority level.");
        perror("set_priority");
    }

    /* Attach to status shared mem area */
    struct guppi_status st;
    rv = guppi_status_attach(&st);
    if (rv!=GUPPI_OK) {
        guppi_error("guppi_dedisp_thread", 
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
    struct psrfits pf;
    pf.sub.dat_freqs = pf.sub.dat_weights =
        pf.sub.dat_offsets = pf.sub.dat_scales = NULL;
    pthread_cleanup_push((void *)guppi_free_psrfits, &pf);

    /* Attach to databuf shared mem */
    struct guppi_databuf *db_in, *db_out;
    db_in = guppi_databuf_attach(args->input_buffer);
    if (db_in==NULL) {
        char msg[256];
        sprintf(msg, "Error attaching to databuf(%d) shared memory.",
                args->input_buffer);
        guppi_error("guppi_dedisp_thread", msg);
        pthread_exit(NULL);
    }
    pthread_cleanup_push((void *)guppi_databuf_detach, db_in);
    db_out = guppi_databuf_attach(args->output_buffer);
    if (db_out==NULL) {
        char msg[256];
        sprintf(msg, "Error attaching to databuf(%d) shared memory.",
                args->output_buffer);
        guppi_error("guppi_dedisp_thread", msg);
        pthread_exit(NULL);
    }
    pthread_cleanup_push((void *)guppi_databuf_detach, db_out);

    /* Loop */
    char *hdr_in=NULL, *hdr_out=NULL;
    struct dedispersion_setup ds;
    char *curdata_out, *dsbuf;
    pthread_cleanup_push((void *)free_dedispersion, &ds);
    pthread_cleanup_push((void *)print_timing_report, &ds);
    int curblock_in=0, curblock_out=0, got_packet_0=0;
    unsigned char *rawdata=NULL;
    float *outdata=NULL;
    int imjd;
    double fmjd, offset;
    int first=1;
    int nblock_int=0, npacket=0, ndrop=0;
    double tsubint=0.0, suboffs=0.0;
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
            guppi_read_obs_params(hdr_in, &gp, &pf);
        else 
            guppi_read_subint_params(hdr_in, &gp, &pf);

        /* Check to see if a new obs started */
        if (gp.packetindex==0) {
            got_packet_0=1;
            guppi_read_obs_params(hdr_in, &gp, &pf);
        }

        /* Get current time */
        //offset = pf.hdr.dt * gp.packetindex * gp.packetsize 
        //    / pf.hdr.nchan / pf.hdr.npol; // Only true for 8-bit data
        const size_t bytes_per_samp = 4;
        offset = pf.hdr.dt * gp.packetindex * gp.packetsize
            / bytes_per_samp / pf.hdr.nchan;
        imjd = pf.hdr.start_day;
        fmjd = (pf.hdr.start_sec + offset) / 86400.0;

        /* Any first-time init stuff */
        if (first) {

            /* Fill in some dedispersion params */
            ds.rf = pf.hdr.fctr;
            ds.bw = pf.hdr.df;
            ds.fft_len = pf.dedisp.fft_len;
            ds.overlap = pf.dedisp.overlap;
            ds.npts_per_block = pf.hdr.nsblk;
            ds.gp = &gp;

            /* Downsample params */
            ds.dsfac = pf.hdr.ds_time_fact;
            ds.npol = pf.hdr.onlyI ? 1 : 4;

            /* Set up freqs */
            int i;
            ds.nchan = pf.hdr.nchan;
            for (i=0; i<ds.nchan; i++)
                ds.freq[i] = ds.rf - pf.hdr.BW/2.0 
                    + ((double)i+0.5)*pf.hdr.df;

            /* Buffers to transfer ds results */
            const size_t dsbuf_size = sizeof(char) * ds.npol 
                * ds.npts_per_block / ds.dsfac;
            cudaMallocHost((void**)&dsbuf, dsbuf_size);

            /* Init dedispersion on GPU */
            ds.dm = pf.hdr.chan_dm;
            printf("DM is %f\n", ds.dm);
            ds.earth_z4 = 0.0;
            init_dedispersion(&ds);

            /* Init downsample */
            init_downsample(&ds);

            /* Clear first time flag */
            first=0;
        }

        /* Setup output data block stuff */
        /* We need to alter various things to trick the psrfits code
         * into thinking this data came from GUPPI1 */
        hdr_out = guppi_databuf_header(db_out, curblock_out);
        curdata_out = (char *)guppi_databuf_data(db_out, curblock_out);
        memcpy(hdr_out, guppi_databuf_header(db_in, curblock_in),
                GUPPI_STATUS_SIZE);
        hputs(hdr_out, "OBS_MODE", "SEARCH");
        hputi4(hdr_out, "ACC_LEN", ds.dsfac);
        hputi4(hdr_out, "DS_TIME", 1);
        hputi4(hdr_out, "NPOL", ds.npol);
        hputi4(hdr_out, "ONLY_I", 0);
        if (ds.npol==1) hputs(hdr_out, "POL_TYPE", "AA+BB");
        hputi4(hdr_out, "BLOCSIZE", ds.npol * ds.nchan * 
                (pf.hdr.nsblk - pf.dedisp.overlap) / ds.dsfac);
        // These are important since it's how search mode psrfits
        // calculates time... 
        hputr8(hdr_out, "TBIN", pf.hdr.dt * ds.dsfac);
        hputi4(hdr_out, "PKTSIZE", ds.npol*ds.nchan); // Spectrum size in bytes
        // TODO this needs to be corrected for overlap.
        // but if we do it here, will 'packet 0' check get screwed up?
        hputi8(hdr_out, "PKTIDX", 
                gp.packetsize*gp.packetindex/ds.dsfac/4/ds.nchan);

        /* Set current time (needed?) */
        ds.imjd = imjd;
        ds.fmjd = fmjd;

        /* Loop over channels in the block */
        unsigned ichan;
        for (ichan=0; ichan<ds.nchan; ichan++) {

            /* Pointer to raw data
             * 4 bytes per sample for 8-bit/2-pol/complex data
             */
            rawdata = (unsigned char *)guppi_databuf_data(db_in, curblock_in) 
                + (size_t)4 * pf.hdr.nsblk * ichan;

            /* Call dedisp fn */
            dedisperse(&ds, ichan, rawdata, outdata);

            /* call downsample */
            downsample(&ds, dsbuf);

            //printf("%d %d\n", ichan, dsbuf[32]);

            // Arrange data into output array in chan, pol, samp order
            // Comes out of GPU in pol, samp order one chan at a time
            // nsblk tells us number of samples per block per chan
            // This assumes 8-bit data.
            // If this transpose is problematic, we could move it
            // to the GPU.
            //const unsigned npts_block = 
            //    (pf.hdr.nsblk-pf.dedisp.overlap)/ds.dsfac;
            const unsigned npts_block = pf.hdr.nsblk / ds.dsfac;
            unsigned isamp;
            for (isamp=0; isamp<npts_block; isamp++) {
                unsigned ipol;
                for (ipol=0; ipol<ds.npol; ipol++) {
                    curdata_out[isamp*ds.nchan*ds.npol + ipol*ds.nchan + ichan] 
                        = dsbuf[ds.npol*isamp+ipol];
                }
            }

        }

        /* Update counters, etc */
        nblock_int++;
        //npacket += gp.n_packets;
        npacket += gp.packets_per_block;
        ndrop += (gp.packets_per_block - gp.n_packets) + gp.n_dropped;
        tsubint = pf.hdr.dt * (npacket - ndrop) * gp.packetsize 
            / pf.hdr.nchan / pf.hdr.npol; // Only true for 8-bit data
        suboffs += offset;
        hputi4(hdr_out, "NPKT", npacket);
        hputi4(hdr_out, "NDROP", ndrop);
        hputi4(hdr_out, "NBLOCK", nblock_int);
        nblock_int=0;
        npacket=0;
        ndrop=0;
        tsubint=0.0;
        suboffs=0.0;

        /* Mark blocks as free/filled */
        guppi_databuf_set_free(db_in, curblock_in);
        guppi_databuf_set_filled(db_out, curblock_out);

        /* Go to next input block */
        curblock_in = (curblock_in + 1) % db_in->n_block;

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

    pthread_cleanup_pop(0); /* Closes print_timing_report */
    pthread_cleanup_pop(0); /* Closes free_dedispersion */
    pthread_cleanup_pop(0); /* Closes guppi_databuf_detach(out) */
    pthread_cleanup_pop(0); /* Closes guppi_databuf_detach(in) */
    pthread_cleanup_pop(0); /* Closes guppi_free_psrfits */
    pthread_cleanup_pop(0); /* Closes guppi_thread_set_finished */
    pthread_cleanup_pop(0); /* Closes set_exit_status */
    pthread_cleanup_pop(0); /* Closes guppi_status_detach */

}
