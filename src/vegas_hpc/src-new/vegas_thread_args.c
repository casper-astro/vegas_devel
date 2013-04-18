#include "vegas_thread_args.h"

void vegas_thread_args_init(struct vegas_thread_args *a) {
    a->priority=0;
    a->finished=0;
    pthread_cond_init(&a->finished_c,NULL);
    pthread_mutex_init(&a->finished_m,NULL);
}

void vegas_thread_args_destroy(struct vegas_thread_args *a) {
    a->finished=1;
    pthread_cond_destroy(&a->finished_c);
    pthread_mutex_destroy(&a->finished_m);
}

void vegas_thread_set_finished(struct vegas_thread_args *a) {
    pthread_mutex_lock(&a->finished_m);
    a->finished=1;
    pthread_cond_broadcast(&a->finished_c);
    pthread_mutex_unlock(&a->finished_m);
}

int vegas_thread_finished(struct vegas_thread_args *a, 
        float timeout_sec) {
    struct timeval now;
    struct timespec twait;
    int rv;
    pthread_mutex_lock(&a->finished_m);
    gettimeofday(&now,NULL);
    twait.tv_sec = now.tv_sec + (int)timeout_sec;
    twait.tv_nsec = now.tv_usec * 1000 + 
        (int)(1e9*(timeout_sec-floor(timeout_sec)));
    if (a->finished==0) 
        rv = pthread_cond_timedwait(&a->finished_c, &a->finished_m, &twait);
    rv = a->finished;
    pthread_mutex_unlock(&a->finished_m);
    return(rv);
}

