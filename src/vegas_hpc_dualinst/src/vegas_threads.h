/* vegas_threads.h
 *
 * Definitions, routines common to 
 * all thread functions.
 */
#ifndef _VEGAS_THREADS_H
#define _VEGAS_THREADS_H

#include "vegas_thread_args.h"

/* SIGINT handling capability */
extern int run;
extern void cc(int sig);

/* Safe lock/unlock functions for status shared mem. */
#define vegas_status_lock_safe(s) \
    pthread_cleanup_push((void *)vegas_status_unlock, s); \
    vegas_status_lock(s);
#define vegas_status_unlock_safe(s) \
    vegas_status_unlock(s); \
    pthread_cleanup_pop(0);

/* Exit handler that updates status buffer */
#ifndef STATUS_KEY
#  define STATUS_KEY "XXXSTAT"
#  define TMP_STATUS_KEY 1
#endif
static void set_exit_status(struct vegas_status *s) {
    vegas_status_lock(s);
    hputs(s->buf, STATUS_KEY, "exiting");
    vegas_status_unlock(s);
}
#if TMP_STATUS_KEY
#  undef STATUS_KEY
#  undef TMP_STATUS_KEY
#endif

#endif
