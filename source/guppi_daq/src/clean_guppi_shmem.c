/* clean_guppi_shmem.c
 *
 * Mark all GUPPI shmem segs for deletion.
 */
#include <stdio.h>
#include <stdlib.h>
#include <sys/ipc.h>
#include <sys/shm.h>
#include <sys/sem.h>
#include <semaphore.h>
#include <fcntl.h>

#include "guppi_status.h"
#include "guppi_databuf.h"
#include "guppi_error.h"

int main(int argc, char *argv[]) {
    int rv,ex=0;

    /* Status shared mem, force unlock first */
    struct guppi_status s;
    sem_unlink(GUPPI_STATUS_SEMID);
    rv = guppi_status_attach(&s);
    if (rv!=GUPPI_OK) {
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
    rv = sem_unlink(GUPPI_STATUS_SEMID);
    if (rv==-1) {
        fprintf(stderr, "Error unlinking status semaphore.\n");
        perror("sem_unlink");
        ex=1;
    }

    /* Databuf shared mem */
    struct guppi_databuf *d=NULL;
    int i = 0;
    for (i=1; i<=2; i++) {
        d = guppi_databuf_attach(i); // Repeat for however many needed ..
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

