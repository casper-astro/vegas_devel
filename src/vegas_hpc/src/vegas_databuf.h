/* vegas_databuf.h
 *
 * Defines shared mem structure for data passing.
 * Includes routines to allocate / attach to shared
 * memory.
 */
#ifndef _VEGAS_DATABUF_H
#define _VEGAS_DATABUF_H

#include <sys/ipc.h>
#include <sys/sem.h>

struct vegas_databuf {
    char data_type[64]; /* Type of data in buffer */
    unsigned int buf_type;  /* GPU_INPUT_BUF or CPU_INPUT_BUF */
    size_t struct_size; /* Size alloced for this struct (bytes) */
    size_t block_size;  /* Size of each data block (bytes) */
    size_t header_size; /* Size of each block header (bytes) */
    size_t index_size;  /* Size of each block's index (bytes) */
    int shmid;          /* ID of this shared mem segment */
    int semid;          /* ID of locking semaphore set */
    int n_block;        /* Number of data blocks in buffer */
};

#define VEGAS_DATABUF_KEY 0x00C62C70

/* union for semaphore ops.  Is this really needed? */
union semun {
    int val;
    struct semid_ds *buf;
    unsigned short *array;
    struct seminfo *__buf;
};

#ifdef NEW_GBT

#define GPU_INPUT_BUF       1
#define CPU_INPUT_BUF       2
#define DISK_INPUT_BUF      3

#define MAX_HEAPS_PER_BLK   4096

// Single element of the index for the GPU or CPU input buffer
struct cpu_gpu_buf_index
{
    unsigned int heap_cntr;
    unsigned int heap_valid;
    double heap_rcvd_mjd;
};

// Single element of the index for the disk input buffer
struct disk_buf_index
{
    unsigned int struct_offset;
    unsigned int array_offset;
};

// The index that sits at the top of each databuf block
struct databuf_index
{
    union {
        unsigned int num_heaps;     //Number of actual heaps in block
        unsigned int num_datasets;  //Number of datasets in block
    };
    
    union {
        unsigned int heap_size;     //Size of a single heap
        unsigned int array_size;    //Size of a single data array
    };
    
    // The actual index
    union {
        struct cpu_gpu_buf_index cpu_gpu_buf[MAX_HEAPS_PER_BLK];
        struct disk_buf_index disk_buf[2*MAX_HEAPS_PER_BLK];
    };
};

#endif


/* Create a new shared mem area with given params.  Returns 
 * pointer to the new area on success, or NULL on error.  Returns
 * error if an existing shmem area exists with the given shmid (or
 * if other errors occured trying to allocate it).
 */
#ifndef NEW_GBT
struct vegas_databuf *vegas_databuf_create(int n_block, size_t block_size,
        int databuf_id);
#else
struct vegas_databuf *vegas_databuf_create(int n_block, size_t block_size,
        int databuf_id, int buf_type);
#endif

/* Return a pointer to a existing shmem segment with given id.
 * Returns error if segment does not exist 
 */
struct vegas_databuf *vegas_databuf_attach(int databuf_id);

/* Detach from shared mem segment */
int vegas_databuf_detach(struct vegas_databuf *d);

/* Clear out either the whole databuf (set all sems to 0, 
 * clear all header blocks) or a single FITS-style
 * header block.
 */
void vegas_databuf_clear(struct vegas_databuf *d);
void vegas_fitsbuf_clear(char *buf);

/* These return pointers to the header, the index or the data area for 
 * the given block_id.
 */
char *vegas_databuf_header(struct vegas_databuf *d, int block_id);
char *vegas_databuf_data(struct vegas_databuf *d, int block_id);
#ifdef NEW_GBT
char *vegas_databuf_index(struct vegas_databuf *d, int block_id);
#endif

/* Returns lock status for given block_id, or total for
 * whole array.
 */
int vegas_databuf_block_status(struct vegas_databuf *d, int block_id);
int vegas_databuf_total_status(struct vegas_databuf *d);

/* Databuf locking functions.  Each block in the buffer
 * can be marked as free or filled.  The "wait" functions
 * block until the specified state happens.  The "set" functions
 * put the buffer in the specified state, returning error if
 * it is already in that state.
 */
int vegas_databuf_wait_filled(struct vegas_databuf *d, int block_id);
int vegas_databuf_set_filled(struct vegas_databuf *d, int block_id);
int vegas_databuf_wait_free(struct vegas_databuf *d, int block_id);
int vegas_databuf_set_free(struct vegas_databuf *d, int block_id);


#endif
