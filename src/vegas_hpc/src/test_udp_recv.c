/* test_udp_recv.c
 *
 * Simple UDP recv tester.  Similar to net_test code.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/types.h>
#include <sys/socket.h>
#include <signal.h>
#include <poll.h>
#include <getopt.h>
#include <errno.h>

#include "vegas_udp.h"
#include "vegas_error.h"

void usage() {
    fprintf(stderr,
            "Usage: vegas_udp [options] sender_hostname\n"
            "Options:\n"
            "  -p n, --port=n    Port number\n"
            "  -h, --help        This message\n"
           );
}

/* control-c handler */
int run=1;
void stop_running(int sig) { run=0; }

int main(int argc, char *argv[]) {

    int rv;
    struct vegas_udp_params p;

    static struct option long_opts[] = {
        {"help",   0, NULL, 'h'},
        {"port",   1, NULL, 'p'},
        {0,0,0,0}
    };
    int opt, opti;
    p.port = 50000;
    p.packet_size=8200; 
    while ((opt=getopt_long(argc,argv,"hp:",long_opts,&opti))!=-1) {
        switch (opt) {
            case 'p':
                p.port = atoi(optarg);
                break;
            default:
            case 'h':
                usage();
                exit(0);
                break;
        }
    }

    /* Default to bee2 if no hostname given */
    if (optind==argc) {
        strcpy(p.sender, "bee2-10");
    } else {
        strcpy(p.sender, argv[optind]);
    }

    /* Init udp params */
    rv = vegas_udp_init(&p);
    if (rv!=VEGAS_OK) { 
        fprintf(stderr, "Error setting up networking\n");
        exit(1);
    }
    printf("sock=%d\n", p.sock);

    int rv2;
    unsigned long long packet_count=0, max_id=0, seq_num;
    struct vegas_udp_packet packet;
    int first=1;
    signal(SIGINT, stop_running);
    printf("Waiting for data (sock=%d).\n", p.sock);
    while (run) {
        rv = vegas_udp_wait(&p);
        if (rv==VEGAS_OK) {
            /* recv data ,etc */
            rv2 = vegas_udp_recv(&p, &packet);
            if (rv2!=VEGAS_OK) {
                if (rv2==VEGAS_ERR_PACKET) { 
                    fprintf(stderr, "unexpected packet size (%zd)\n",
                            packet.packet_size);
                } else if (rv2==VEGAS_ERR_SYS) {
                    if (errno!=EAGAIN) {
                        printf("sock=%d\n", p.sock);
                        perror("recv");
                        exit(1);
                    }
                } else {
                    fprintf(stderr, "Unknown error = %d\n", rv2);
                }
            } else {
                if (first) { 
                    printf("Receiving (packet_size=%d).\n", (int)p.packet_size);
                    first=0;
                } 
                packet_count++;
                seq_num = vegas_udp_packet_seq_num(&packet);
                if (seq_num>max_id) { max_id=seq_num; }
            }
        } else if (rv==VEGAS_TIMEOUT) {
            if (first==0) { run=0; }
        } else {
            if (run) {
                perror("poll");
                vegas_udp_close(&p);
                exit(1);
            } else {
                printf("Caught SIGINT, exiting.\n");
            }
        }
    }

    printf("Received %lld packets, dropped %lld (%.3e)\n",
            packet_count, max_id+1-packet_count, 
            (double)(max_id+1-packet_count)/(double)(max_id+1));



    vegas_udp_close(&p);
    exit(0);
}
