/* guppi_status.c
 *
 * Implementation of the status routines described 
 * in guppi_status.h
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/ipc.h>
#include <sys/shm.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <semaphore.h>

#include "guppi_status.h"
#include "guppi_error.h"

int guppi_status_attach(struct guppi_status *s) {

    /* Get shared mem id (creating it if necessary) */
    s->shmid = shmget(GUPPI_STATUS_KEY, GUPPI_STATUS_SIZE, 0666 | IPC_CREAT);
    if (s->shmid==-1) { 
        guppi_error("guppi_status_attach", "shmget error");
        return(GUPPI_ERR_SYS);
    }

    /* Now attach to the segment */
    s->buf = shmat(s->shmid, NULL, 0);
    if (s->buf == (void *)-1) {
        printf("shmid=%d\n", s->shmid);
        guppi_error("guppi_status_attach", "shmat error");
        return(GUPPI_ERR_SYS);
    }

    /* Get the locking semaphore.
     * Final arg (1) means create in unlocked state (0=locked).
     */
    mode_t old_umask = umask(0);
    s->lock = sem_open(GUPPI_STATUS_SEMID, O_CREAT, 0666, 1);
    umask(old_umask);
    if (s->lock==SEM_FAILED) {
        guppi_error("guppi_status_attach", "sem_open");
        return(GUPPI_ERR_SYS);
    }

    /* Init buffer if needed */
    guppi_status_chkinit(s);

    return(GUPPI_OK);
}

int guppi_status_detach(struct guppi_status *s) {
    int rv = shmdt(s->buf);
    if (rv!=0) {
        guppi_error("guppi_status_detach", "shmdt error");
        return(GUPPI_ERR_SYS);
    }
    s->buf = NULL;
    return(GUPPI_OK);
}

/* TODO: put in some (long, ~few sec) timeout */
int guppi_status_lock(struct guppi_status *s) {
    return(sem_wait(s->lock));
}

int guppi_status_unlock(struct guppi_status *s) {
    return(sem_post(s->lock));
}

/* Return pointer to END key */
char *guppi_find_end(char *buf) {
    /* Loop over 80 byte cards */
    int offs;
    char *out=NULL;
    for (offs=0; offs<GUPPI_STATUS_SIZE; offs+=GUPPI_STATUS_CARD) {
        if (strncmp(&buf[offs], "END", 3)==0) { out=&buf[offs]; break; }
    }
    return(out);
}

/* So far, just checks for existence of "END" in the proper spot */
void guppi_status_chkinit(struct guppi_status *s) {

    /* Lock */
    guppi_status_lock(s);

    /* If no END, clear it out */
    if (guppi_find_end(s->buf)==NULL) {
        /* Zero bufer */
        memset(s->buf, 0, GUPPI_STATUS_SIZE);
        /* Fill first card w/ spaces */
        memset(s->buf, ' ', GUPPI_STATUS_CARD);
        /* add END */
        strncpy(s->buf, "END", 3);
    }

    /* Unlock */
    guppi_status_unlock(s);
}

/* Clear out guppi status buf */
void guppi_status_clear(struct guppi_status *s) {

    /* Lock */
    guppi_status_lock(s);

    /* Zero bufer */
    memset(s->buf, 0, GUPPI_STATUS_SIZE);
    /* Fill first card w/ spaces */
    memset(s->buf, ' ', GUPPI_STATUS_CARD);
    /* add END */
    strncpy(s->buf, "END", 3);

    /* Unlock */
    guppi_status_unlock(s);
}
