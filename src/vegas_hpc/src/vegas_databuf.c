/* vegas_databuf.c
 *
 * Routines for creating and accessing main data transfer
 * buffer in shared memory.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/types.h>
#include <sys/ipc.h>
#include <sys/shm.h>
#include <sys/sem.h>
#include <errno.h>
#include <time.h>

#include "fitshead.h"
#include "vegas_status.h"
#include "vegas_databuf.h"
#include "vegas_error.h"


struct vegas_databuf *vegas_databuf_create(int n_block, size_t block_size,
        int databuf_id, int buf_type) {

    /* Calc databuf size */
    const size_t header_size = VEGAS_STATUS_SIZE;
    size_t struct_size = sizeof(struct vegas_databuf);
    struct_size = 8192 * (1 + struct_size/8192); /* round up */
    size_t index_size = sizeof(struct databuf_index);
    size_t databuf_size = (block_size+header_size+index_size) * n_block + struct_size;

    /* Get shared memory block, error if it already exists */
    int shmid;
    shmid = shmget(VEGAS_DATABUF_KEY + databuf_id - 1, 
            databuf_size, 0666 | IPC_CREAT | IPC_EXCL);
    if (shmid==-1) {
        vegas_error("vegas_databuf_create", "shmget error");
        return(NULL);
    }

    /* Attach */
    struct vegas_databuf *d;
    d = shmat(shmid, NULL, 0);
    if (d==(void *)-1) {
        vegas_error("vegas_databuf_create", "shmat error");
        return(NULL);
    }

    /* Try to lock in memory */
    int rv = shmctl(shmid, SHM_LOCK, NULL);
    if (rv==-1) {
        vegas_error("vegas_databuf_create", "Error locking shared memory.");
        perror("shmctl");
    }

    /* Zero out memory */
    memset(d, 0, databuf_size);

    /* Fill params into databuf */
    int i;
    char end_key[81];
    memset(end_key, ' ', 80);
    strncpy(end_key, "END", 3);
    end_key[80]='\0';
    d->shmid = shmid;
    d->semid = 0;
    d->n_block = n_block;
    d->databuf_size = databuf_size;
    d->struct_size = struct_size;
    d->block_size = block_size;
    d->header_size = header_size;
    d->index_size = index_size;
    sprintf(d->data_type, "unknown");
    d->buf_type = buf_type;

    for (i=0; i<n_block; i++) { 
        memcpy(vegas_databuf_header(d,i), end_key, 80); 
    }

    /* Get semaphores set up.
       If the disk buffer (type=3), we make 1024 semaphores, as the blocks
       may be resized later. */
    if(buf_type == DISK_INPUT_BUF)
        d->semid = semget(VEGAS_DATABUF_KEY + databuf_id - 1, MAX_BLKS_PER_BUF, 0666 | IPC_CREAT);
    else
        d->semid = semget(VEGAS_DATABUF_KEY + databuf_id - 1, n_block, 0666 | IPC_CREAT);

    if (d->semid==-1) { 
        vegas_error("vegas_databuf_create", "semget error");
        return(NULL);
    }

    /* Init semaphores to 0 */
    union semun arg;

    if(buf_type == DISK_INPUT_BUF)
    {
        arg.array = (unsigned short *)malloc(sizeof(unsigned short)*MAX_BLKS_PER_BUF);
        memset(arg.array, 0, sizeof(unsigned short)*MAX_BLKS_PER_BUF);
    }
    else
    {
        arg.array = (unsigned short *)malloc(sizeof(unsigned short)*n_block);
        memset(arg.array, 0, sizeof(unsigned short)*n_block);
    }

    rv = semctl(d->semid, 0, SETALL, arg);
    free(arg.array);

    return(d);
}


/** 
 * Resizes the blocks within the specified databuf. The number of blocks
 * are automatically changed, so that the total buffer size remains constant.
 */
void vegas_conf_databuf_size(struct vegas_databuf *d, size_t new_block_size)
{

    /* Calculate number of data blocks that can fit into the existing buffer */
    int new_n_block = (d->databuf_size - d->struct_size) / (new_block_size + d->header_size + d->index_size);
    
    /* Make sure that there won't be more data blocks than semaphores */
    if(new_n_block > MAX_BLKS_PER_BUF)
    {
        printf("Warning: the disk buffer contains more than %d blocks. Only %d blocks will be used\n",
                MAX_BLKS_PER_BUF, MAX_BLKS_PER_BUF);
        new_n_block = MAX_BLKS_PER_BUF;
    }

    /* Fill params into databuf */
    d->n_block = new_n_block;
    d->block_size = new_block_size;

    return;
}


int vegas_databuf_detach(struct vegas_databuf *d) {
    int rv = shmdt(d);
    if (rv!=0) {
        vegas_error("vegas_status_detach", "shmdt error");
        return(VEGAS_ERR_SYS);
    }
    return(VEGAS_OK);
}

void vegas_databuf_clear(struct vegas_databuf *d) {

    /* Zero out semaphores */
    union semun arg;
    if(d->buf_type == DISK_INPUT_BUF)
    {
      arg.array = (unsigned short *)malloc(sizeof(unsigned short)*MAX_BLKS_PER_BUF);
      memset(arg.array, 0, sizeof(unsigned short)*MAX_BLKS_PER_BUF);
    }
    else
    {
      arg.array = (unsigned short *)malloc(sizeof(unsigned short)*d->n_block);
      memset(arg.array, 0, sizeof(unsigned short)*d->n_block);
    }

    semctl(d->semid, 0, SETALL, arg);
    free(arg.array);

    /* Clear all headers */
    int i;
    for (i=0; i<d->n_block; i++) {
        vegas_fitsbuf_clear(vegas_databuf_header(d, i));
    }

}

void vegas_fitsbuf_clear(char *buf) {
    char *end, *ptr;
    end = ksearch(buf, "END");
    if (end!=NULL) {
        for (ptr=buf; ptr<=end; ptr+=80) memset(ptr, ' ', 80);
    }
    memset(buf, ' ' , 80);
    strncpy(buf, "END", 3);
}

#ifndef NEW_GBT
char *vegas_databuf_header(struct vegas_databuf *d, int block_id) {
    return((char *)d + d->struct_size + block_id*d->header_size);
}

char *vegas_databuf_data(struct vegas_databuf *d, int block_id) {
    return((char *)d + d->struct_size + d->n_block*d->header_size
            + block_id*d->block_size);
}

#else

char *vegas_databuf_header(struct vegas_databuf *d, int block_id) {
    return((char *)d + d->struct_size + block_id*d->header_size);
}

char *vegas_databuf_index(struct vegas_databuf *d, int block_id) {
    return((char *)d + d->struct_size + d->n_block*d->header_size
            + block_id*d->index_size);
}

char *vegas_databuf_data(struct vegas_databuf *d, int block_id) {
    return((char *)d + d->struct_size + d->n_block*d->header_size
            + d->n_block*d->index_size + block_id*d->block_size);
}
#endif

struct vegas_databuf *vegas_databuf_attach(int databuf_id) {

    /* Get shmid */
    int shmid;
    shmid = shmget(VEGAS_DATABUF_KEY + databuf_id - 1, 0, 0666);
    if (shmid==-1) {
        // Doesn't exist, exit quietly otherwise complain
        if (errno!=ENOENT)
            vegas_error("vegas_databuf_attach", "shmget error");
        return(NULL);
    }

    /* Attach */
    struct vegas_databuf *d;
    d = shmat(shmid, NULL, 0);
    if (d==(void *)-1) {
        vegas_error("vegas_databuf_attach", "shmat error");
        return(NULL);
    }

    return(d);

}

int vegas_databuf_block_status(struct vegas_databuf *d, int block_id) {
    return(semctl(d->semid, block_id, GETVAL));
}

int vegas_databuf_total_status(struct vegas_databuf *d) {

    /* Get all values at once */
    union semun arg;
    arg.array = (unsigned short *)malloc(sizeof(unsigned short)*MAX_BLKS_PER_BUF);
    
    memset(arg.array, 0, sizeof(unsigned short)*MAX_BLKS_PER_BUF);
    semctl(d->semid, 0, GETALL, arg);
    int i,tot=0;
    for (i=0; i<d->n_block; i++) tot+=arg.array[i];
    free(arg.array);
    return(tot);

}

int vegas_databuf_wait_free(struct vegas_databuf *d, int block_id) {
    int rv;
    struct sembuf op;
    op.sem_num = block_id;
    op.sem_op = 0;
    op.sem_flg = 0;
    struct timespec timeout;
    timeout.tv_sec = 0;
    timeout.tv_nsec = 250000000;
    rv = semtimedop(d->semid, &op, 1, &timeout);
    if (rv==-1) { 
        if (errno==EAGAIN) return(VEGAS_TIMEOUT);
        if (errno==EINTR) return(VEGAS_ERR_SYS);
        vegas_error("vegas_databuf_wait_free", "semop error");
        perror("semop");
        return(VEGAS_ERR_SYS);
    }
    return(0);
}

int vegas_databuf_wait_filled(struct vegas_databuf *d, int block_id) {
    /* This needs to wait for the semval of the given block
     * to become > 0, but NOT immediately decrement it to 0.
     * Probably do this by giving an array of semops, since
     * (afaik) the whole array happens atomically:
     * step 1: wait for val=1 then decrement (semop=-1)
     * step 2: increment by 1 (semop=1)
     */
    int rv;
    struct sembuf op[2];
    op[0].sem_num = op[1].sem_num = block_id;
    op[0].sem_flg = op[1].sem_flg = 0;
    op[0].sem_op = -1;
    op[1].sem_op = 1;
    struct timespec timeout;
    timeout.tv_sec = 0;
    timeout.tv_nsec = 250000000;
    rv = semtimedop(d->semid, op, 2, &timeout);
    if (rv==-1) { 
        if (errno==EAGAIN) return(VEGAS_TIMEOUT);
        // Don't complain on a signal interruption
        if (errno==EINTR) return(VEGAS_ERR_SYS);
        vegas_error("vegas_databuf_wait_filled", "semop error");
        perror("semop");
        return(VEGAS_ERR_SYS);
    }
    return(0);
}

int vegas_databuf_set_free(struct vegas_databuf *d, int block_id) {
    /* This function should always succeed regardless of the current
     * state of the specified databuf.  So we use semctl (not semop) to set
     * the value to zero.
     */
    int rv;
    union semun arg;
    arg.val = 0;
    rv = semctl(d->semid, block_id, SETVAL, arg);
    if (rv==-1) { 
        vegas_error("vegas_databuf_set_free", "semctl error");
        return(VEGAS_ERR_SYS);
    }
    return(0);
}

int vegas_databuf_set_filled(struct vegas_databuf *d, int block_id) {
    /* This function should always succeed regardless of the current
     * state of the specified databuf.  So we use semctl (not semop) to set
     * the value to one.
     */
    int rv;
    union semun arg;
    arg.val = 1;
    rv = semctl(d->semid, block_id, SETVAL, arg);
    if (rv==-1) { 
        vegas_error("vegas_databuf_set_filled", "semctl error");
        return(VEGAS_ERR_SYS);
    }
    return(0);
}
