/* check_vegas_databuf.c
 *
 * Basic prog to test dstabuf shared mem routines.
 */
#include <stdio.h>
#include <stdlib.h>
#include <getopt.h>
#include <string.h>

#include "fitshead.h"
#include "vegas_error.h"
#include "vegas_status.h"
#include "vegas_databuf.h"
#include "vegas_defines.h"

void usage() { 
    fprintf(stderr, 
            "Usage: check_vegas_databuf [options]\n"
            "Options:\n"
            "  -h, --help\n"
            "  -q, --quiet\n"
            "  -c, --create\n"
            "  -i n, --id=n  (1)\n"
            "  -s n, --size=n (32768)\n"
            "  -n n, --nblock=n (24)\n"
            );
}

int main(int argc, char *argv[]) {

    /* Loop over cmd line to fill in params */
    static struct option long_opts[] = {
        {"help",   0, NULL, 'h'},
        {"quiet",  0, NULL, 'q'},
        {"create", 0, NULL, 'c'},
        {"id",     1, NULL, 'i'},
        {"size",   1, NULL, 's'},
        {"nblock", 1, NULL, 'n'},
#ifdef NEW_GBT
        {"type", 1, NULL, 't'},
#endif
        {0,0,0,0}
    };
    int opt,opti;
    int quiet=0;
    int create=0;
    int db_id=1;
    int blocksize = 32768;
    int nblock = 24;
    int type = 1;
    while ((opt=getopt_long(argc,argv,"hqci:s:n:t:",long_opts,&opti))!=-1) {
        switch (opt) {
            case 'c':
                create=1;
                break;
            case 'q':
                quiet=1;
                break;
            case 'i':
                db_id = atoi(optarg);
                break;
            case 's':
                blocksize = atoi(optarg);
                break;
            case 'n':
                nblock = atoi(optarg);
                break;
#ifdef NEW_GBT
            case 't':
                type = atoi(optarg);
                break;
#endif
            case 'h':
            default:
                usage();
                exit(0);
                break;
        }
    }

    /* Create mem if asked, otherwise attach */
    struct vegas_databuf *db=NULL;
    if (create) { 
#ifndef NEW_GBT
        db = vegas_databuf_create(nblock, blocksize*1024*1024, db_id);
#else
        db = vegas_databuf_create(nblock, blocksize*1024, db_id, type);
#endif
        if (db==NULL) {
            fprintf(stderr, "Error creating databuf %d (may already exist).\n",
                    db_id);
            exit(1);
        }
    } else {
        db = vegas_databuf_attach(db_id);
        if (db==NULL) { 
            fprintf(stderr, 
                    "Error attaching to databuf %d (may not exist).\n",
                    db_id);
            exit(1);
        }
    }

    /* Print basic info */
    printf("databuf %d stats:\n", db_id);
    printf("  shmid=%d\n", db->shmid);
    printf("  semid=%d\n", db->semid);
    printf("  n_block=%d\n", db->n_block);
    printf("  struct_size=%zd\n", db->struct_size);
    printf("  block_size=%zd\n", db->block_size);
    printf("  header_size=%zd\n\n", db->header_size);

    /* loop over blocks */
    int i;
    char buf[81];
    char *hdr, *ptr, *hend;
    for (i=0; i<db->n_block; i++) {
        printf("block %d status=%d\n", i, 
                vegas_databuf_block_status(db, i));
        hdr = vegas_databuf_header(db, i);
        hend = ksearch(hdr, "END");
        if (hend==NULL) {
            printf("header not initialized\n");
        } else {
            hend += 80;
            printf("header:\n");
            for (ptr=hdr; ptr<hend; ptr+=80) {
                strncpy(buf, ptr, 80);
                buf[79]='\0';
                printf("%s\n", buf);
            }
        }

    }

    exit(0);
}
