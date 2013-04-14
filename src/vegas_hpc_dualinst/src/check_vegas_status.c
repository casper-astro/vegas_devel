/* check_vegas_status.c
 *
 * Basic prog to test status shared mem routines.
 */
#include <stdio.h>
#include <stdlib.h>
#include <getopt.h>

#include "fitshead.h"
#include "vegas_error.h"
#include "vegas_status.h"

static struct vegas_status *get_status_buffer(int instance_id)
{
    int rv;
    static int last_used_instance_id = -1;
    static struct vegas_status s;

    instance_id &= 0x3f;

    if(last_used_instance_id != instance_id) {
      rv = vegas_status_attach(instance_id, &s);
      if (rv!=VEGAS_OK) {
          // Should "never" happen
          fprintf(stderr, "Error connecting to status buffer instance %d.\n",
              instance_id);
          perror("vegas_status_attach");
          exit(1);
      }
    }

    return &s;
}

int main(int argc, char *argv[]) {
    int instance_id = 0;
    struct vegas_status *s;

    /* Loop over cmd line to fill in params */
    static struct option long_opts[] = {
        {"key",    1, NULL, 'k'},
        {"get",    1, NULL, 'g'},
        {"string", 1, NULL, 's'},
        {"float",  1, NULL, 'f'},
        {"double", 1, NULL, 'd'},
        {"int",    1, NULL, 'i'},
        {"quiet",  0, NULL, 'q'},
        {"clear",  0, NULL, 'C'},
        {"del",    0, NULL, 'D'},
        {"instance", 1, NULL, 'I'},
        {0,0,0,0}
    };
    int opt,opti;
    char *key=NULL;
    float flttmp;
    double dbltmp;
    int inttmp;
    int quiet=0, clear=0;
    while ((opt=getopt_long(argc,argv,"k:g:s:f:d:i:qCDI:",long_opts,&opti))!=-1) {
        switch (opt) {
            case 'I':
                instance_id = atoi(optarg);
                break;
            case 'k':
                key = optarg;
                break;
            case 'g':
                s = get_status_buffer(instance_id);
                vegas_status_lock(s);
                hgetr8(s->buf, optarg, &dbltmp);
                vegas_status_unlock(s);
                printf("%g\n", dbltmp);
                break;
            case 's':
                if (key) {
                    s = get_status_buffer(instance_id);
                    vegas_status_lock(s);
                    hputs(s->buf, key, optarg);
                    vegas_status_unlock(s);
                }
                break;
            case 'f':
                flttmp = atof(optarg);
                if (key) {
                    s = get_status_buffer(instance_id);
                    vegas_status_lock(s);
                    hputr4(s->buf, key, flttmp);
                    vegas_status_unlock(s);
                }
                break;
            case 'd':
                dbltmp = atof(optarg);
                if (key) {
                    s = get_status_buffer(instance_id);
                    vegas_status_lock(s);
                    hputr8(s->buf, key, dbltmp);
                    vegas_status_unlock(s);
                }
                break;
            case 'i':
                inttmp = atoi(optarg);
                if (key) {
                    s = get_status_buffer(instance_id);
                    vegas_status_lock(s);
                    hputi4(s->buf, key, inttmp);
                    vegas_status_unlock(s);
                }
                break;
            case 'D':
                if (key) {
                    s = get_status_buffer(instance_id);
                    vegas_status_lock(s);
                    hdel(s->buf, key);
                    vegas_status_unlock(s);
                }
                break;
            case 'C':
                clear=1;
                break;
            case 'q':
                quiet=1;
                break;
            default:
                break;
        }
    }

    s = get_status_buffer(instance_id);

    /* If not quiet, print out buffer */
    if (!quiet) { 
        printf(s->buf); printf("\n"); 
    }

    vegas_status_unlock(s);

    if (clear) 
        vegas_status_clear(s);

    exit(0);
}
