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
            "Usage: guppi_daq_fold [options] sender_hostname\n"
            "Options:\n"
            "  -h, --help        This message\n"
           );
}

/* Thread declarations */
void *guppi_net_thread(void *args);
void *guppi_fold_thread(void *args);
void *guppi_psrfits_thread(void *args);
void *guppi_null_thread(void *args);

int main(int argc, char *argv[]) {

    static struct option long_opts[] = {
        {"help",   0, NULL, 'h'},
        {"null",   0, NULL, 'n'},
        {0,0,0,0}
    };
    int use_null_thread = 0;
    int opt, opti;
    while ((opt=getopt_long(argc,argv,"hn",long_opts,&opti))!=-1) {
        switch (opt) {
            case 'n':
                use_null_thread = 1;
                break;
            default:
            case 'h':
                usage();
                exit(0);
                break;
        }
    }

    /* Data buffer ids */
    struct guppi_thread_args net_args, fold_args, disk_args;
    guppi_thread_args_init(&net_args);
    guppi_thread_args_init(&fold_args);
    guppi_thread_args_init(&disk_args);
    net_args.output_buffer = 1;
    fold_args.input_buffer = net_args.output_buffer;
    fold_args.output_buffer = 2;
    disk_args.input_buffer = fold_args.output_buffer;
    //fold_args.priority = 10;
    //net_args.priority = -10;

    /* Init shared mem */
    struct guppi_status stat;
    struct guppi_databuf *dbuf_net=NULL, *dbuf_fold=NULL;
    int rv = guppi_status_attach(&stat);
    if (rv!=GUPPI_OK) {
        fprintf(stderr, "Error connecting to guppi_status\n");
        exit(1);
    }

    dbuf_net = guppi_databuf_attach(net_args.output_buffer);
    if (dbuf_net==NULL) {
        fprintf(stderr, "Error connecting to guppi_databuf\n");
        exit(1);
    }
    guppi_databuf_clear(dbuf_net);

    dbuf_fold = guppi_databuf_attach(fold_args.output_buffer);
    if (dbuf_fold==NULL) {
        fprintf(stderr, "Error connecting to guppi_databuf\n");
        exit(1);
    }
    guppi_databuf_clear(dbuf_fold);


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

    /* Launch fold thread */
    pthread_t fold_thread_id;
    rv = pthread_create(&fold_thread_id, NULL, guppi_fold_thread, 
            (void *)&fold_args);
    if (rv) { 
        fprintf(stderr, "Error creating fold thread.\n");
        perror("pthread_create");
        exit(1);
    }

    /* Launch psrfits/null thread */
    pthread_t disk_thread_id=0;
    if (use_null_thread)
        rv = pthread_create(&disk_thread_id, NULL, guppi_null_thread,
                (void *)&disk_args);
    else
        rv = pthread_create(&disk_thread_id, NULL, guppi_psrfits_thread,
                (void *)&disk_args);
    if (rv) { 
        fprintf(stderr, "Error creating psrfits thread.\n");
        perror("pthread_create");
        exit(1);
    }


    /* Alt loop, wait for run=0 */
    while (run) {
        sleep(1); 
        if (disk_args.finished) run=0;
    }

    /* Clean up */
    pthread_cancel(fold_thread_id);
    pthread_cancel(net_thread_id);
    pthread_cancel(disk_thread_id);
    pthread_kill(fold_thread_id,SIGINT);
    pthread_kill(net_thread_id,SIGINT);
    pthread_kill(disk_thread_id,SIGINT);
    pthread_join(net_thread_id,NULL);
    printf("Joined net thread\n"); fflush(stdout);
    pthread_join(fold_thread_id,NULL);
    printf("Joined fold thread\n"); fflush(stdout);
    pthread_join(disk_thread_id,NULL);
    printf("Joined disk thread\n"); fflush(stdout);

    guppi_thread_args_destroy(&net_args);
    guppi_thread_args_destroy(&fold_args);
    guppi_thread_args_destroy(&disk_args);

    exit(0);
}
