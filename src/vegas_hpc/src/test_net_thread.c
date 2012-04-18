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
#include "guppi_defines.h"

void usage() {
    fprintf(stderr,
            "Usage: test_net_thread [options] [sender_hostname]\n"
            "Default hostname: bee2-10\n"
            "Options:\n"
            "  -h, --help        This message\n"
            "  -d, --disk        Write raw data to disk (default no)\n"
            "  -o, --only_net    Run only guppi_net_thread\n"
           );
}

/* Thread declarations */
void *guppi_net_thread(void *_up);
void *guppi_rawdisk_thread(void *args);
void *guppi_null_thread(void *args);

int main(int argc, char *argv[]) {

    static struct option long_opts[] = {
        {"help",    0, NULL, 'h'},
        {"disk",    0, NULL, 'd'},
        {"only_net",0, NULL, 'o'},
        {0,0,0,0}
    };
    int opt, opti;
    int disk=0, only_net=0;
    while ((opt=getopt_long(argc,argv,"hdo",long_opts,&opti))!=-1) {
        switch (opt) {
            case 'd':
                disk=1;
                break;
            case 'o':
                only_net=1;
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

    /* Launch raw disk (or null) thread */
    struct guppi_thread_args null_args;
    guppi_thread_args_init(&null_args);
    null_args.input_buffer = net_args.output_buffer;
    pthread_t disk_thread_id=0;
    if (only_net==0) {
        if (disk)
            rv = pthread_create(&disk_thread_id, NULL, guppi_rawdisk_thread, 
                    (void *)&null_args);
        else
            rv = pthread_create(&disk_thread_id, NULL, guppi_null_thread, 
                    (void *)&null_args);
        if (rv) { 
            fprintf(stderr, "Error creating null thread.\n");
            perror("pthread_create");
            exit(1);
        }
    }

    /* Wait for end */
    while (run) { sleep(1); }
    if (disk_thread_id) pthread_cancel(disk_thread_id);
    pthread_cancel(net_thread_id);
    if (disk_thread_id) pthread_kill(disk_thread_id,SIGINT);
    pthread_kill(net_thread_id,SIGINT);
    pthread_join(net_thread_id,NULL);
    printf("Joined net thread\n"); fflush(stdout); fflush(stderr);
    if (disk_thread_id) {
        pthread_join(disk_thread_id,NULL);
        printf("Joined disk thread\n"); fflush(stdout); fflush(stderr);
    }

    guppi_thread_args_destroy(&null_args);

    exit(0);
}
