/* guppi_daq.c
 *
 * The main GUPPI program.
 */
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>
#include <pthread.h>
#include <sys/types.h>
#include <sys/socket.h>
#include <signal.h>
#include <poll.h>
#include <getopt.h>
#include <errno.h>

#include "guppi_error.h"
#include "guppi_status.h"
#include "guppi_databuf.h"
#include "guppi_params.h"
#include "guppi_thread_main.h"
#include "guppi_defines.h"

/* Thread declarations */
void *guppi_net_thread(void *args);
#if FITS_TYPE == PSRFITS
void *guppi_psrfits_thread(void *args);
#else
void *guppi_sdfits_thread(void *args);
#endif

#ifdef RAW_DISK
void *guppi_rawdisk_thread(void *args);
#endif

#ifdef FAKE_NET
void *guppi_fake_net_thread(void *args);
#endif


int main(int argc, char *argv[]) {

    /* thread args */
    struct guppi_thread_args net_args, disk_args;
    guppi_thread_args_init(&net_args);
    guppi_thread_args_init(&disk_args);
    net_args.output_buffer = 1;
    disk_args.input_buffer = net_args.output_buffer;

    /* Init shared mem */
    struct guppi_status stat;
    struct guppi_databuf *dbuf=NULL;
    int rv = guppi_status_attach(&stat);
    if (rv!=GUPPI_OK) {
        fprintf(stderr, "Error connecting to guppi_status\n");
        exit(1);
    }
    dbuf = guppi_databuf_attach(net_args.output_buffer);
    /* If attach fails, first try to create the databuf */
    if (dbuf==NULL) 
#ifdef NEW_GBT
        dbuf = guppi_databuf_create(24, 32*1024*1024, net_args.output_buffer, CPU_INPUT_BUF);
#else
        dbuf = guppi_databuf_create(24, 32*1024*1024, net_args.output_buffer);
#endif
    /* If that also fails, exit */
    if (dbuf==NULL) {
        fprintf(stderr, "Error connecting to guppi_databuf\n");
        exit(1);
    }

    guppi_databuf_clear(dbuf);

    signal(SIGINT, cc);

    /* Launch net thread */
    pthread_t net_thread_id;
#ifdef FAKE_NET
    rv = pthread_create(&net_thread_id, NULL, guppi_fake_net_thread,
            (void *)&net_args);
#else
    rv = pthread_create(&net_thread_id, NULL, guppi_net_thread,
            (void *)&net_args);
#endif
    if (rv) { 
        fprintf(stderr, "Error creating net thread.\n");
        perror("pthread_create");
        exit(1);
    }

    /* Launch RAW_DISK thread, SDFITS disk thread, or PSRFITS disk thread */
    pthread_t disk_thread_id;
#ifdef RAW_DISK
    rv = pthread_create(&disk_thread_id, NULL, guppi_rawdisk_thread, 
        (void *)&disk_args);
#elif FITS_TYPE == PSRFITS
    rv = pthread_create(&disk_thread_id, NULL, guppi_psrfits_thread, 
        (void *)&disk_args);
#elif FITS_TYPE == SDFITS
    rv = pthread_create(&disk_thread_id, NULL, guppi_sdfits_thread, 
        (void *)&disk_args);
#endif
    if (rv) { 
        fprintf(stderr, "Error creating disk thread.\n");
        perror("pthread_create");
        exit(1);
    }

    /* Wait for end */
    run=1;
    while (run) { 
        sleep(1); 
        if (disk_args.finished) run=0;
    }
    pthread_cancel(disk_thread_id);
    pthread_cancel(net_thread_id);
    pthread_kill(disk_thread_id,SIGINT);
    pthread_kill(net_thread_id,SIGINT);
    pthread_join(net_thread_id,NULL);
    printf("Joined net thread\n"); fflush(stdout);
    pthread_join(disk_thread_id,NULL);
    printf("Joined disk thread\n"); fflush(stdout);

    guppi_thread_args_destroy(&disk_args);

    exit(0);
}
