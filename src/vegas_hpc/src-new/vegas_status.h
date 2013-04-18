/* vegas_status.h
 *
 * Routines dealing with the vegas status shared memory
 * segment.  Info is passed through this segment using 
 * a FITS-like keyword=value syntax.
 */
#ifndef _VEGAS_STATUS_H
#define _VEGAS_STATUS_H

#include <semaphore.h>

#include "vegas_params.h"

#define VEGAS_STATUS_KEY 0x01001840
#define VEGAS_STATUS_SEMID "/vegas_status"
#define VEGAS_STATUS_SIZE (2880*64) // FITS-style buffer
#define VEGAS_STATUS_CARD 80 // Size of each FITS "card"

#define VEGAS_LOCK 1
#define VEGAS_NOLOCK 0

/* Structure describes status memory area */
struct vegas_status {
    int shmid;   /* Shared memory segment id */
    sem_t *lock; /* POSIX semaphore descriptor for locking */
    char *buf;   /* Pointer to data area */
};

/* Return a pointer to the status shared mem area, 
 * creating it if it doesn't exist.  Attaches/creates 
 * lock semaphore as well.  Returns nonzero on error.
 */
int vegas_status_attach(struct vegas_status *s);

/* Detach from shared mem segment */
int vegas_status_detach(struct vegas_status *s); 

/* Lock/unlock the status buffer.  vegas_status_lock() will wait for
 * the buffer to become unlocked.  Return non-zero on errors.
 */
int vegas_status_lock(struct vegas_status *s);
int vegas_status_unlock(struct vegas_status *s);

/* Check the buffer for appropriate formatting (existence of "END").
 * If not found, zero it out and add END.
 */
void vegas_status_chkinit(struct vegas_status *s);

/* Clear out whole buffer */
void vegas_status_clear(struct vegas_status *s);

#endif
