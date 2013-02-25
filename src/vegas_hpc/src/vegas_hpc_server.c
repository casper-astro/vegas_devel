/* vegas_hpc_server.c
 *
 * Persistent process that will await commands on a FIFO
 * and spawn datataking threads as appropriate.  Meant for
 * communication w/ VEGAS manager, etc.
 */
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>
#include <pthread.h>
#include <sys/types.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <sys/prctl.h>
#include <fcntl.h>
#include <signal.h>
#include <poll.h>
#include <getopt.h>
#include <errno.h>
#include <time.h>

#include "fitshead.h"
#include "vegas_error.h"
#include "vegas_status.h"
#include "vegas_databuf.h"
#include "vegas_params.h"

#include "vegas_thread_main.h"

#define vegas_DAQ_CONTROL "/tmp/vegas_daq_control"

void usage() {
    fprintf(stderr,
            "Usage: vegas_daq_server [options]\n"
            "Options:\n"
            "  -h, --help        This message\n"
           );
}

/* Override "usual" SIGINT stuff */
int srv_run=1;
void srv_cc(int sig) { srv_run=0; run=0; }
void srv_quit(int sig) { srv_run=0; }

/* Thread declarations */
void *vegas_net_thread(void *args);
void *vegas_null_thread(void *args);
void *vegas_pfb_thread(void *args);
void *vegas_accum_thread(void *args);
void *vegas_rawdisk_thread(void *args);

/* Useful thread functions */

int check_thread_exit(struct vegas_thread_args *args, int nthread) {
    int i, rv=0;
    for (i=0; i<nthread; i++) 
        rv += args[i].finished;
    return(rv);
}

void init_hbw_mode(struct vegas_thread_args *args, int *nthread) {
    vegas_thread_args_init(&args[0]); // net
    vegas_thread_args_init(&args[1]); // accum
    args[0].output_buffer = 2;
    args[1].input_buffer = args[0].output_buffer;
    args[1].output_buffer = 3;
    *nthread = 2;
}

void init_lbw_mode(struct vegas_thread_args *args, int *nthread) {
    vegas_thread_args_init(&args[0]); // net
    vegas_thread_args_init(&args[1]); // pfb
    vegas_thread_args_init(&args[2]); // accum
    args[0].output_buffer = 1;
    args[1].input_buffer = args[0].output_buffer;
    args[1].output_buffer = 2;
    args[2].input_buffer = args[1].output_buffer;
    args[2].output_buffer = 3;
    *nthread = 3;
}

void init_monitor_mode(struct vegas_thread_args *args, int *nthread) {
    vegas_thread_args_init(&args[0]); // net
    vegas_thread_args_init(&args[1]); // null
    args[0].output_buffer = 1;
    args[1].input_buffer = args[0].output_buffer;
    *nthread = 2;
}

void start_hbw_mode(struct vegas_thread_args *args, pthread_t *ids) {
    // TODO error checking...
    int rv;
    rv = pthread_create(&ids[0], NULL, vegas_net_thread, (void*)&args[0]);
    rv = pthread_create(&ids[1], NULL, vegas_accum_thread, (void*)&args[1]);
}

void start_lbw_mode(struct vegas_thread_args *args, pthread_t *ids) {
    // TODO error checking...
    int rv;
    rv = pthread_create(&ids[0], NULL, vegas_net_thread, (void*)&args[0]);
    rv = pthread_create(&ids[1], NULL, vegas_pfb_thread, (void*)&args[1]);
    rv = pthread_create(&ids[2], NULL, vegas_accum_thread, (void*)&args[2]);
}

void start_monitor_mode(struct vegas_thread_args *args, pthread_t *ids) {
    // TODO error checking...
    int rv;
    rv = pthread_create(&ids[0], NULL, vegas_net_thread, (void*)&args[0]);
    rv = pthread_create(&ids[1], NULL, vegas_null_thread, (void*)&args[1]);
}

void stop_threads(struct vegas_thread_args *args, pthread_t *ids,
        unsigned nthread) {
    unsigned i;
    for (i=0; i<nthread; i++) pthread_cancel(ids[i]);
    for (i=0; i<nthread; i++) pthread_kill(ids[i], SIGINT);
    for (i=0; i<nthread; i++) pthread_join(ids[i], NULL);
    for (i=0; i<nthread; i++) vegas_thread_args_destroy(&args[i]);
    for (i=0; i<nthread; i++) ids[i] = 0;
}

int main(int argc, char *argv[]) {

    static struct option long_opts[] = {
        {"help",   0, NULL, 'h'},
        {0,0,0,0}
    };
    int opt, opti;
    while ((opt=getopt_long(argc,argv,"h",long_opts,&opti))!=-1) {
        switch (opt) {
            default:
            case 'h':
                usage();
                exit(0);
                break;
        }
    }
    
    prctl(PR_SET_PDEATHSIG,SIGTERM); /* Ensure that if parent (the manager) dies, to kill this child too. */

    /* Create FIFO */
    int rv = mkfifo(vegas_DAQ_CONTROL, 0666);
    if (rv!=0 && errno!=EEXIST) {
        fprintf(stderr, "vegas_daq_server: Error creating control fifo\n");
        perror("mkfifo");
        exit(1);
    }

    /* Open command FIFO for read */
#define MAX_CMD_LEN 1024
    char cmd[MAX_CMD_LEN];
    int command_fifo;
    command_fifo = open(vegas_DAQ_CONTROL, O_RDONLY | O_NONBLOCK);
    if (command_fifo<0) {
        fprintf(stderr, "vegas_daq_server: Error opening control fifo\n");
        perror("open");
        exit(1);
    }

    /* Attach to shared memory buffers */
    struct vegas_status stat;
    struct vegas_databuf *dbuf_net=NULL, *dbuf_pfb=NULL, *dbuf_acc=NULL;
    rv = vegas_status_attach(&stat);
    const int netbuf_id = 1;
    const int pfbbuf_id = 2;
    const int accbuf_id = 3;
    if (rv!=VEGAS_OK) {
        fprintf(stderr, "Error connecting to vegas_status\n");
        exit(1);
    }
    dbuf_net = vegas_databuf_attach(netbuf_id);
    if (dbuf_net==NULL) {
        fprintf(stderr, "Error connecting to vegas_databuf (raw net)\n");
        exit(1);
    }
    vegas_databuf_clear(dbuf_net);
    dbuf_pfb = vegas_databuf_attach(pfbbuf_id);
    if (dbuf_pfb==NULL) {
        fprintf(stderr, "Error connecting to vegas_databuf (accum input)\n");
        exit(1);
    }
    vegas_databuf_clear(dbuf_pfb);

    dbuf_acc = vegas_databuf_attach(accbuf_id);
    if (dbuf_acc==NULL) {
        fprintf(stderr, "Error connecting to vegas_databuf (accum output)\n");
        exit(1);
    }
    vegas_databuf_clear(dbuf_acc);

    /* Thread setup */
#define MAX_THREAD 8
    int i;
    int nthread_cur = 0;
    struct vegas_thread_args args[MAX_THREAD];
    pthread_t thread_id[MAX_THREAD];
    for (i=0; i<MAX_THREAD; i++) thread_id[i] = 0;

    /* Print start time for logs */
    time_t curtime = time(NULL);
    char tmp[256];
    printf("\nvegas_daq_server started at %s", ctime_r(&curtime,tmp));
    fflush(stdout);

    /* hmm.. keep this old signal stuff?? */
    run=1;
    srv_run=1;
    signal(SIGINT, srv_cc);
    signal(SIGTERM, srv_quit);

    /* Loop over recv'd commands, process them */
    int cmd_wait=1;
    while (cmd_wait && srv_run) {

        // Check to see if threads have exited, if so, stop them
        if (check_thread_exit(args, nthread_cur)) {
            run = 0;
            stop_threads(args, thread_id, nthread_cur);
            nthread_cur = 0;
        }

        // Heartbeat, status update
        time_t curtime;
        char timestr[32];
        char *ctmp;
        time(&curtime);
        ctime_r(&curtime, timestr);
        ctmp = strchr(timestr, '\n');
        if (ctmp!=NULL) { *ctmp = '\0'; } else { timestr[0]='\0'; }
        vegas_status_lock(&stat);
        hputs(stat.buf, "DAQPULSE", timestr);
        hputs(stat.buf, "DAQSTATE", nthread_cur==0 ? "stopped" : "running");
        vegas_status_unlock(&stat);

        // Flush any status/error/etc for logfiles
        fflush(stdout);
        fflush(stderr);

        // Wait for data on fifo
        struct pollfd pfd;
        pfd.fd = command_fifo;
        pfd.events = POLLIN;
        rv = poll(&pfd, 1, 1000);
        if (rv==0) { continue; }
        else if (rv<0) {
            if (errno!=EINTR) perror("poll");
            continue;
        }

        // If we got POLLHUP, it means the other side closed its
        // connection.  Close and reopen the FIFO to clear this
        // condition.  Is there a better/recommended way to do this?
        if (pfd.revents==POLLHUP) { 
            close(command_fifo);
            command_fifo = open(vegas_DAQ_CONTROL, O_RDONLY | O_NONBLOCK);
            if (command_fifo<0) {
                fprintf(stderr, 
                        "vegas_daq_server: Error opening control fifo\n");
                perror("open");
                break;
            }
            continue;
        }

        // Read the command
        memset(cmd, 0, MAX_CMD_LEN);
        rv = read(command_fifo, cmd, MAX_CMD_LEN-1);
        if (rv==0) { continue; }
        else if (rv<0) {
            if (errno==EAGAIN) { continue; }
            else { perror("read");  continue; }
        } 

        // Truncate at newline
        // TODO: allow multiple commands in one read?
        char *ptr = strchr(cmd, '\n');
        if (ptr!=NULL) *ptr='\0'; 

        // Process the command 
        if (strncasecmp(cmd,"QUIT",MAX_CMD_LEN)==0) {
            // Exit program
            printf("Exit\n");
            run = 0;
            stop_threads(args, thread_id, nthread_cur);
            cmd_wait=0;
            continue;
        } 
        
        else if (strncasecmp(cmd,"START",MAX_CMD_LEN)==0 ||
                strncasecmp(cmd,"MONITOR",MAX_CMD_LEN)==0) {
            // Start observations
            // TODO : decide how to behave if observations are running
            printf("Start observations\n");

            if (nthread_cur>0) {
                printf("  observations already running!\n");
            } else {

                // Figure out which mode to start
                char obs_mode[32];
                if (strncasecmp(cmd,"START",MAX_CMD_LEN)==0) {
                    vegas_status_lock(&stat);
                    vegas_read_obs_mode(stat.buf, obs_mode);
                    vegas_status_unlock(&stat);
                } else {
                    strncpy(obs_mode, cmd, 32);
                }
                printf("  obs_mode = %s\n", obs_mode);

                // Clear out data bufs
                vegas_databuf_clear(dbuf_net);
                vegas_databuf_clear(dbuf_pfb);
                vegas_databuf_clear(dbuf_acc);


                // Do it
                run = 1;
                if (strncasecmp(obs_mode, "HBW", 4)==0) {
                    hputs(stat.buf, "BW_MODE", "high");
                    hputs(stat.buf, "SWVER", "1.4");
                    init_hbw_mode(args, &nthread_cur);
                    start_hbw_mode(args, thread_id);
                } else if (strncasecmp(obs_mode, "LBW", 4)==0) {
                    hputs(stat.buf, "BW_MODE", "low");
                    hputs(stat.buf, "SWVER", "1.4");

                    init_lbw_mode(args, &nthread_cur);
                    start_lbw_mode(args, thread_id);
                } else if (strncasecmp(obs_mode, "MONITOR", 8)==0) {
                    init_monitor_mode(args, &nthread_cur);
                    start_monitor_mode(args, thread_id);
                } else {
                    printf("  unrecognized obs_mode!\n");
                }

            }

        } 
        
        else if (strncasecmp(cmd,"STOP",MAX_CMD_LEN)==0) {
            // Stop observations
            printf("Stop observations\n");
            run = 0;
            stop_threads(args, thread_id, nthread_cur);
            nthread_cur = 0;
        } 
        
        else {
            // Unknown command
            printf("Unrecognized command '%s'\n", cmd);
        }
    }

    /* Stop any running threads */
    run = 0;
    stop_threads(args, thread_id, nthread_cur);

    if (command_fifo>0) close(command_fifo);

    vegas_status_lock(&stat);
    hputs(stat.buf, "DAQSTATE", "exiting");
    vegas_status_unlock(&stat);

    curtime = time(NULL);
    printf("vegas_daq_server exiting cleanly at %s\n", ctime_r(&curtime,tmp));

    fflush(stdout);
    fflush(stderr);

    /* TODO: remove FIFO */

    exit(0);
}
