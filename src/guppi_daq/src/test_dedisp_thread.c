/* test_net_thread.c
 *
 * Test run net thread.
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

void usage() {
    fprintf(stderr,
            "Usage: test_dedisp_thread [options] [sender_hostname]\n"
            "Default hostname: bee2_10\n"
            "Options:\n"
            "  -h, --help        This message\n"
            "  -d, --disk        Write raw data to disk (default no)\n"
            "  -D, --ds          Downsample instead of fold\n"
           );
}

/* Thread declarations */
void *guppi_net_thread(void *_up);
void *guppi_dedisp_thread(void *args);
void *guppi_dedisp_ds_thread(void *args);

int main(int argc, char *argv[]) {

    static struct option long_opts[] = {
        {"help",   0, NULL, 'h'},
        {"disk",   0, NULL, 'd'},
        {"ds",     0, NULL, 'D'},
        {0,0,0,0}
    };
    int opt, opti;
    int disk=0;
    int ds=0;
    while ((opt=getopt_long(argc,argv,"hdD",long_opts,&opti))!=-1) {
        switch (opt) {
            case 'd':
                disk=1;
                break;
            case 'D':
                ds=1;
                break;
            default:
            case 'h':
                usage();
                exit(0);
                break;
        }
    }

    /* Net thread args */
    struct guppi_thread_args net_args;
    guppi_thread_args_init(&net_args);
    net_args.output_buffer = 1;

    /* Init shared mem */
    struct guppi_status stat;
    struct guppi_databuf *dbuf1=NULL, *dbuf2=NULL;
    int rv = guppi_status_attach(&stat);
    if (rv!=GUPPI_OK) {
        fprintf(stderr, "Error connecting to guppi_status\n");
        exit(1);
    }

    /* Clear data buf 1 */
    dbuf1 = guppi_databuf_attach(net_args.output_buffer);
    /* If attach fails, first try to create the databuf */
    if (dbuf1==NULL) 
        dbuf1= guppi_databuf_create(24, 32*1024*1024, net_args.output_buffer);
    /* If that also fails, exit */
    if (dbuf1==NULL) {
        fprintf(stderr, "Error connecting to guppi_databuf\n");
        exit(1);
    }
    guppi_databuf_clear(dbuf1);

    /* Clear data buf 2 */
    dbuf2 = guppi_databuf_attach(2);
    /* If attach fails, first try to create the databuf */
    if (dbuf2==NULL) 
        dbuf2= guppi_databuf_create(24, 32*1024*1024, net_args.output_buffer);
    /* If that also fails, exit */
    if (dbuf2==NULL) {
        fprintf(stderr, "Error connecting to guppi_databuf\n");
        exit(1);
    }
    guppi_databuf_clear(dbuf2);

    run=1;
    signal(SIGINT, cc);

    /* Launch net thread */
    pthread_t net_thread_id;
    rv = pthread_create(&net_thread_id, NULL, guppi_net_thread,
            (void *)&net_args);
    if (rv) { 
        fprintf(stderr, "Error creating net thread.\n");
        perror("pthread_create");
        exit(1);
    }

    /* Launch dedisp thread */
    struct guppi_thread_args dedisp_args;
    guppi_thread_args_init(&dedisp_args);
    dedisp_args.input_buffer = net_args.output_buffer;
    dedisp_args.output_buffer = 2;
    pthread_t dedisp_thread_id;
    if (ds)
        rv = pthread_create(&dedisp_thread_id, NULL, guppi_dedisp_ds_thread, 
                (void *)&dedisp_args);
    else
        rv = pthread_create(&dedisp_thread_id, NULL, guppi_dedisp_thread, 
                (void *)&dedisp_args);
    if (rv) { 
        fprintf(stderr, "Error creating dedisp thread.\n");
        perror("pthread_create");
        exit(1);
    }

    /* Wait for end */
    while (run) { sleep(1); }
    pthread_cancel(dedisp_thread_id);
    pthread_cancel(net_thread_id);
    pthread_kill(dedisp_thread_id,SIGINT);
    pthread_kill(net_thread_id,SIGINT);
    pthread_join(net_thread_id,NULL);
    printf("Joined net thread\n"); fflush(stdout); fflush(stderr);
    pthread_join(dedisp_thread_id,NULL);
    printf("Joined dedisp thread\n"); fflush(stdout); fflush(stderr);

    guppi_thread_args_destroy(&dedisp_args);

    //cudaThreadExit(); // XXX??

    exit(0);
}
