/* vegas_status.c
 *
 * Implementation of the status routines described 
 * in vegas_status.h
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

#include "vegas_status.h"
#include "vegas_error.h"
#include "vegas_ipckey.h"

/* Returns the vegas status (POSIX) semaphore name. */
const char * vegas_status_semname(int instance_id)
{
    static char semid[NAME_MAX-4] = {'\0'};
    int length_remaining = NAME_MAX-4;
    char *s;
    // Lazy init
    if(semid[0] == '\0') {
        const char * envstr = getenv("VEGAS_STATUS_SEMNAME");
        if(envstr) {
            strncpy(semid, envstr, length_remaining);
            semid[length_remaining-1] = '\0';
        } else {
            envstr = getenv("VEGAS_KEYFILE");
            if(!envstr) {
                envstr = getenv("HOME");
                if(!envstr) {
                    envstr = "/tmp";
                }
            }
            strncpy(semid, envstr, length_remaining);
            semid[length_remaining-1] = '\0';
            // Convert all but the leading / to _
            s = semid + 1;
            while((s = strchr(s, '/'))) {
              *s = '_';
            }
            length_remaining -= strlen(semid);
            if(length_remaining > 0) {
                snprintf(semid+strlen(semid), length_remaining,
                    "_vegas_status_%d", instance_id&0x3f);
            }
        }
#ifdef VEGAS_VERBOSE
        fprintf(stderr, "using vegas status semaphore '%s'\n", semid);
#endif
    }
    return semid;
}

int vegas_status_attach(int instance_id, struct vegas_status *s) {
    instance_id &= 0x3f;
    s->instance_id = instance_id;

    /* Get shared mem id (creating it if necessary) */
    key_t key = vegas_status_key(instance_id);
    if(key == VEGAS_KEY_ERROR) {
        vegas_error("vegas_status_attach", "vegas_status_key error");
        return(0);
    }
    s->shmid = shmget(key, VEGAS_STATUS_SIZE, 0666 | IPC_CREAT);
    if (s->shmid==-1) { 
        vegas_error("vegas_status_attach", "shmget error");
        return(VEGAS_ERR_SYS);
    }

    /* Now attach to the segment */
    s->buf = shmat(s->shmid, NULL, 0);
    if (s->buf == (void *)-1) {
        printf("shmid=%d\n", s->shmid);
        vegas_error("vegas_status_attach", "shmat error");
        return(VEGAS_ERR_SYS);
    }

    /* Get the locking semaphore.
     * Final arg (1) means create in unlocked state (0=locked).
     */
    mode_t old_umask = umask(0);
    s->lock = sem_open(vegas_status_semname(instance_id), O_CREAT, 0666, 1);
    umask(old_umask);
    if (s->lock==SEM_FAILED) {
        vegas_error("vegas_status_attach", "sem_open");
        return(VEGAS_ERR_SYS);
    }

    /* Init buffer if needed */
    vegas_status_chkinit(s);

    return(VEGAS_OK);
}

int vegas_status_detach(struct vegas_status *s) {
    int rv = shmdt(s->buf);
    if (rv!=0) {
        vegas_error("vegas_status_detach", "shmdt error");
        return(VEGAS_ERR_SYS);
    }
    s->buf = NULL;
    return(VEGAS_OK);
}

/* TODO: put in some (long, ~few sec) timeout */
int vegas_status_lock(struct vegas_status *s) {
    return(sem_wait(s->lock));
}

int vegas_status_unlock(struct vegas_status *s) {
    return(sem_post(s->lock));
}

/* Return pointer to END key */
char *vegas_find_end(char *buf) {
    /* Loop over 80 byte cards */
    int offs;
    char *out=NULL;
    for (offs=0; offs<VEGAS_STATUS_SIZE; offs+=VEGAS_STATUS_CARD) {
        if (strncmp(&buf[offs], "END", 3)==0) { out=&buf[offs]; break; }
    }
    return(out);
}

/* So far, just checks for existence of "END" in the proper spot */
void vegas_status_chkinit(struct vegas_status *s) {

    int instance_id = -1;
    int semval;
    int retval;
    retval = sem_getvalue(s->lock,&semval);
    if (retval) {
        vegas_error("vegas_status_chkinit", "sem_getvalue failed");

    }
    if (semval == 0) {
        printf("Found vegas status semaphore locked in vegas_status_chkinit. releasing\n");
        vegas_status_unlock(s);
    }

    /* Lock */
    vegas_status_lock(s);

    /* If no END, clear it out */
    if (vegas_find_end(s->buf)==NULL) {
        /* Zero bufer */
        memset(s->buf, 0, VEGAS_STATUS_SIZE);
        /* Fill first card w/ spaces */
        memset(s->buf, ' ', VEGAS_STATUS_CARD);
        /* add END */
        strncpy(s->buf, "END", 3);
        // Add INSTANCE record
        hputi4(s->buf, "INSTANCE", s->instance_id);
    } else {
        // Check INSTANCE record
        if(!hgeti4(s->buf, "INSTANCE", &instance_id)) {
            // No INSTANCE record, so add one
            hputi4(s->buf, "INSTANCE", s->instance_id);
        } else if(instance_id != s->instance_id) {
            // Print warning message
            fprintf(stderr,
                "Existing INSTANCE value %d != desired value %d\n",
                instance_id, s->instance_id);
            // Fix it (Really?  Why did this condition exist anyway?)
            hputi4(s->buf, "INSTANCE", s->instance_id);
        }
    }

    /* Unlock */
    vegas_status_unlock(s);
}

/* Clear out vegas status buf */
void vegas_status_clear(struct vegas_status *s) {

    int semval;
    int retval;
    retval = sem_getvalue(s->lock,&semval);
    if (retval) {
        vegas_error("vegas_status_clear", "sem_getvalue failed");
    }
    if (semval == 0) {
        printf("Found vegas status semaphore locked in vegas_status_clear. releasing\n");
        vegas_status_unlock(s);
    }

    /* Lock */
    vegas_status_lock(s);

    /* Zero bufer */
    memset(s->buf, 0, VEGAS_STATUS_SIZE);
    /* Fill first card w/ spaces */
    memset(s->buf, ' ', VEGAS_STATUS_CARD);
    /* add END */
    strncpy(s->buf, "END", 3);

    hputi4(s->buf, "INSTANCE", s->instance_id);

    /* Unlock */
    vegas_status_unlock(s);
}
