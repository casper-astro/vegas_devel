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

#include "vegas_status.h"
#include "vegas_databuf.h"
#include "vegas_error.h"

int main(int argc, char *argv[]) {
    int rv,ex=0;

    /* Status shared mem, force unlock first */
    struct vegas_status s;
    sem_unlink(VEGAS_STATUS_SEMID);
    rv = vegas_status_attach(&s);
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
    rv = sem_unlink(VEGAS_STATUS_SEMID);
    if (rv==-1) {
        fprintf(stderr, "Error unlinking status semaphore.\n");
        perror("sem_unlink");
        ex=1;
    }

    /* Databuf shared mem */
    struct vegas_databuf *d=NULL;
    int i = 0;
    for (i=1; i<=2; i++) {
        d = vegas_databuf_attach(i); // Repeat for however many needed ..
        if (d==NULL) continue;
        if (d->semid) { 
            rv = semctl(d->semid, 0, IPC_RMID); 
            if (rv==-1) {
                fprintf(stderr, "Error removing databuf semaphore\n");
                perror("semctl");
                ex=1;
            }
        }
        rv = shmctl(d->shmid, IPC_RMID, NULL);
        if (rv==-1) {
            fprintf(stderr, "Error deleting databuf segment.\n");
            perror("shmctl");
            ex=1;
        }
    }

    exit(ex);
}

