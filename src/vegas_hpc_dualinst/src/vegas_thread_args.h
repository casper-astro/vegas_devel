#ifndef _VEGAS_THREAD_ARGS_H
#define _VEGAS_THREAD_ARGS_H
/* Generic thread args type with input/output buffer
 * id numbers.  Not all threads have both a input and a
 * output.
 */
#include <pthread.h>
#include <sys/time.h>
#include <math.h>
struct vegas_thread_args {
    int instance_id;
    int input_buffer;
    int output_buffer;
    int priority;
    int finished;
    pthread_cond_t finished_c;
    pthread_mutex_t finished_m;
};
void vegas_thread_args_init(struct vegas_thread_args *a, int instance_id);
void vegas_thread_args_destroy(struct vegas_thread_args *a);
void vegas_thread_set_finished(struct vegas_thread_args *a);
int vegas_thread_finished(struct vegas_thread_args *a, 
        float timeout_sec);
#endif
