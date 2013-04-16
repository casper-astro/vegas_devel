/* clean_vegas_shmem.c
 *
 * Mark all VEGAS shmem segs for deletion.
 */
#include <stdio.h>
#include <stdlib.h>
#include <sys/ipc.h>
#include <sys/shm.h>
#include <sys/sem.h>
#include <semaphore.h>
#include <fcntl.h>
#include <getopt.h>

#include "vegas_status.h"
#include "vegas_databuf.h"
#include "vegas_error.h"

int main(int argc, char *argv[]) {
    int rv,ex=0;
    int instance_id = 0;
    int delete_status = 0;
    int opt;

    /* Loop over cmd line to fill in params */
    // "-d" as command line argument deletes status memory and semaphore.
    // Otherwise, it is simply re-initialized.
    static struct option long_opts[] = {
        {"instance", 1, NULL, 'I'},
        {0,0,0,0}
    };

    while ((opt=getopt_long(argc,argv,"I:",long_opts,NULL))!=-1) {
        switch (opt) {
            case 'I':
                instance_id = atoi(optarg);
                break;
            case '?': // Command line parsing error
            default:
                exit(1);
                break;
        }
    }

    /* Status shared mem, force unlock first */
    struct vegas_status s;
    const char *semname = vegas_status_semname(instance_id);
    sem_unlink(semname);
    rv = vegas_status_attach(instance_id, &s);
    if (rv!=VEGAS_OK) {
        fprintf(stderr, "Error connecting to status shared mem.\n");
        perror(NULL);
        exit(1);
    }
    rv = shmctl(s.shmid, IPC_RMID, NULL);
    if (rv==-1) {
        fprintf(stderr, "Error deleting status segment.\n");
        perror("shmctl");
        ex=1;
    }
    rv = sem_unlink(semname);
    if (rv==-1) {
        fprintf(stderr, "Error unlinking status semaphore.\n");
        perror("sem_unlink");
        ex=1;
    }

    /* Databuf shared mem */
    struct vegas_databuf *d=NULL;
    int i = 0;
    for (i=1; i<=2; i++) {
        d = vegas_databuf_attach(instance_id, i); // Repeat for however many needed ..
        if (d==NULL) continue;
        if (d->semid) { 
            rv = semctl(d->semid, 0, IPC_RMID); 
            if (rv==-1) {
                fprintf(stderr, "Error removing databuf semaphore %u\n", d->semid);
                perror("semctl");
                ex=1;
            }
        }
        rv = shmctl(d->shmid, IPC_RMID, NULL);
        if (rv==-1) {
            fprintf(stderr, "Error deleting databuf segment %u.\n", d->shmid);
            perror("shmctl");
            ex=1;
        }
    }

    exit(ex);
}

