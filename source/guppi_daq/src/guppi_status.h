/* guppi_status.h
 *
 * Routines dealing with the guppi status shared memory
 * segment.  Info is passed through this segment using 
 * a FITS-like keyword=value syntax.
 */
#ifndef _GUPPI_STATUS_H
#define _GUPPI_STATUS_H

#include <semaphore.h>

#include "guppi_params.h"

#define GUPPI_STATUS_KEY 16783408
#define GUPPI_STATUS_SEMID "/guppi_status"
#define GUPPI_STATUS_SIZE (2880*64) // FITS-style buffer
#define GUPPI_STATUS_CARD 80 // Size of each FITS "card"

#define GUPPI_LOCK 1
#define GUPPI_NOLOCK 0

/* Structure describes status memory area */
struct guppi_status {
    int shmid;   /* Shared memory segment id */
    sem_t *lock; /* POSIX semaphore descriptor for locking */
    char *buf;   /* Pointer to data area */
};

/* Return a pointer to the status shared mem area, 
 * creating it if it doesn't exist.  Attaches/creates 
 * lock semaphore as well.  Returns nonzero on error.
 */
int guppi_status_attach(struct guppi_status *s);

/* Detach from shared mem segment */
int guppi_status_detach(struct guppi_status *s); 

/* Lock/unlock the status buffer.  guppi_status_lock() will wait for
 * the buffer to become unlocked.  Return non-zero on errors.
 */
int guppi_status_lock(struct guppi_status *s);
int guppi_status_unlock(struct guppi_status *s);

/* Check the buffer for appropriate formatting (existence of "END").
 * If not found, zero it out and add END.
 */
void guppi_status_chkinit(struct guppi_status *s);

/* Clear out whole buffer */
void guppi_status_clear(struct guppi_status *s);

#endif
