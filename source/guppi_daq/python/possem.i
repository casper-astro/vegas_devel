%module possem
%{
#include <semaphore.h>
%}

// Treat a mode_t as an unsigned integer
typedef int mode_t;

// From <bits/fcntl.h>
#define O_CREAT            0100 /* not fcntl */
#define O_EXCL             0200 /* not fcntl */

//sem_t *sem_open(const char *name, int oflag);
sem_t *sem_open(const char *name, int oflag,
                mode_t mode, unsigned int value);
int sem_post(sem_t *sem);
int sem_wait(sem_t *sem);
int sem_close(sem_t *sem);
