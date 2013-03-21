/* guppi_udp.c
 *
 * UDP implementations.
 */
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>
#include <sys/types.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <netdb.h>
#include <fcntl.h>
#include <poll.h>

#include "guppi_udp.h"
#include "guppi_databuf.h"
#include "guppi_error.h"

int guppi_udp_init(struct guppi_udp_params *p) {

    /* Resolve sender hostname */
    struct addrinfo hints;
    struct addrinfo *result, *rp;
    memset(&hints, 0, sizeof(struct addrinfo));
    hints.ai_family = AF_INET;
    hints.ai_socktype = SOCK_DGRAM;
    int rv = getaddrinfo(p->sender, NULL, &hints, &result);
    if (rv!=0) { 
        guppi_error("guppi_udp_init", "getaddrinfo failed");
        return(GUPPI_ERR_SYS);
    }

    /* Set up socket */
    p->sock = socket(PF_INET, SOCK_DGRAM, 0);
    if (p->sock==-1) { 
        guppi_error("guppi_udp_init", "socket error");
        freeaddrinfo(result);
        return(GUPPI_ERR_SYS);
    }

    /* bind to local address */
    struct sockaddr_in local_ip;
    local_ip.sin_family =  AF_INET;
    local_ip.sin_port = htons(p->port);
    local_ip.sin_addr.s_addr = INADDR_ANY;
    rv = bind(p->sock, (struct sockaddr *)&local_ip, sizeof(local_ip));
    if (rv==-1) {
        guppi_error("guppi_udp_init", "bind");
        return(GUPPI_ERR_SYS);
    }

    /* Set up socket to recv only from sender */
    for (rp=result; rp!=NULL; rp=rp->ai_next) {
        if (connect(p->sock, rp->ai_addr, rp->ai_addrlen)==0) { break; }
    }
    if (rp==NULL) { 
        guppi_error("guppi_udp_init", "connect error");
        close(p->sock); 
        freeaddrinfo(result);
        return(GUPPI_ERR_SYS);
    }
    memcpy(&p->sender_addr, rp, sizeof(struct addrinfo));
    freeaddrinfo(result);

    /* Non-blocking recv */
    fcntl(p->sock, F_SETFL, O_NONBLOCK);

    /* Increase recv buffer for this sock */
    int bufsize = 128*1024*1024;
    socklen_t ss = sizeof(int);
    rv = setsockopt(p->sock, SOL_SOCKET, SO_RCVBUF, &bufsize, sizeof(int));
    if (rv<0) { 
        guppi_error("guppi_udp_init", "Error setting rcvbuf size.");
        perror("setsockopt");
    } 
    rv = getsockopt(p->sock, SOL_SOCKET, SO_RCVBUF, &bufsize, &ss); 
    if (0 && rv==0) { 
        printf("guppi_udp_init: SO_RCVBUF=%d\n", bufsize);
    }

    /* Poll command */
    p->pfd.fd = p->sock;
    p->pfd.events = POLLIN;

    return(GUPPI_OK);
}

int guppi_udp_wait(struct guppi_udp_params *p) {
    int rv = poll(&p->pfd, 1, 1000); /* Timeout 1sec */
    if (rv==1) { return(GUPPI_OK); } /* Data ready */
    else if (rv==0) { return(GUPPI_TIMEOUT); } /* Timed out */
    else { return(GUPPI_ERR_SYS); }  /* Other error */
}

int guppi_udp_recv(struct guppi_udp_params *p, struct guppi_udp_packet *b) {
    int rv = recv(p->sock, b->data, GUPPI_MAX_PACKET_SIZE, 0);
    b->packet_size = rv;
    if (rv==-1) { return(GUPPI_ERR_SYS); }
    else if (p->packet_size) {
        if (rv!=p->packet_size) { return(GUPPI_ERR_PACKET); }
        else { return(GUPPI_OK); }
    } else { 
        p->packet_size = rv;
        return(GUPPI_OK); 
    }
}

unsigned long long change_endian64(const unsigned long long *d) {
    unsigned long long tmp;
    char *in=(char *)d, *out=(char *)&tmp;
    int i;
    for (i=0; i<8; i++) {
        out[i] = in[7-i];
    }
    return(tmp);
}

unsigned long long guppi_udp_packet_seq_num(const struct guppi_udp_packet *p) {
    return(change_endian64((unsigned long long *)(p->data)));
}

#define PACKET_SIZE_ORIG ((size_t)8208)
#define PACKET_SIZE_SHORT ((size_t)544)
#define PACKET_SIZE_1SFA ((size_t)8224)
#define PACKET_SIZE_1SFA_OLD ((size_t)8160)
#define PACKET_SIZE_FAST4K ((size_t)4128)

size_t guppi_udp_packet_datasize(size_t packet_size) {
    /* Special case for the new "1SFA" packets, which have an extra
     * 16 bytes at the end reserved for future use.  All other guppi
     * packets have 8 bytes index at the front, and 8 bytes error
     * flags at the end.
     * NOTE: This represents the "full" packet output size...
     */
    if (packet_size==PACKET_SIZE_1SFA) // 1SFA packet size
        return((size_t)8192);
    else if (packet_size==PACKET_SIZE_FAST4K) 
        return((size_t)4096);
    else if (packet_size==PACKET_SIZE_SHORT) 
        //return((size_t)256);
        return((size_t)512);
    else              
        return(packet_size - 2*sizeof(unsigned long long));
}

char *guppi_udp_packet_data(const struct guppi_udp_packet *p) {
    /* This is valid for all guppi packet formats */
    return((char *)(p->data) + sizeof(unsigned long long));
}

unsigned long long guppi_udp_packet_flags(const struct guppi_udp_packet *p) {
    return(*(unsigned long long *)((char *)(p->data) 
                + p->packet_size - sizeof(unsigned long long)));
}

/* Copy the data portion of a guppi udp packet to the given output
 * address.  This function takes care of expanding out the 
 * "missing" channels in 1SFA packets.
 */
void guppi_udp_packet_data_copy(char *out, const struct guppi_udp_packet *p) {
    if (p->packet_size==PACKET_SIZE_1SFA_OLD) {
        /* Expand out, leaving space for missing data.  So far only 
         * need to deal with 4k-channel case of 2 spectra per packet.
         * May need to be updated in the future if 1SFA works with 
         * different numbers of channels.
         *
         * TODO: Update 5/12/2009, newer 1SFA modes always will have full 
         * data contents, and the old 4k ones never really worked, so
         * this code can probably be deleted.
         */
        const size_t pad = 16;
        const size_t spec_data_size = 4096 - 2*pad;
        memset(out, 0, pad);
        memcpy(out + pad, guppi_udp_packet_data(p), spec_data_size);
        memset(out + pad + spec_data_size, 0, 2*pad);
        memcpy(out + pad + spec_data_size + pad + pad, 
                guppi_udp_packet_data(p) + spec_data_size, 
                spec_data_size);
        memset(out + pad + spec_data_size + pad
                + pad + spec_data_size, 0, pad);
    } else {
        /* Packet has full data, just do a memcpy */
        memcpy(out, guppi_udp_packet_data(p), 
                guppi_udp_packet_datasize(p->packet_size));
    }
}

size_t parkes_udp_packet_datasize(size_t packet_size) {
    return(packet_size - sizeof(unsigned long long));
}

void parkes_to_guppi(struct guppi_udp_packet *b, const int acc_len, 
        const int npol, const int nchan) {

    /* Convert IBOB clock count to packet count.
     * This assumes 2 samples per IBOB clock, and that
     * acc_len is the actual accumulation length (=reg_acclen+1).
     */
    const unsigned int counts_per_packet = (nchan/2) * acc_len;
    unsigned long long *packet_idx = (unsigned long long *)b->data;
    (*packet_idx) = change_endian64(packet_idx);
    (*packet_idx) /= counts_per_packet;
    (*packet_idx) = change_endian64(packet_idx);

    /* Reorder from the 2-pol Parkes ordering */
    int i;
    char tmp[GUPPI_MAX_PACKET_SIZE];
    char *pol0, *pol1, *pol2, *pol3, *in;
    in = b->data + sizeof(long long);
    if (npol==2) {
        pol0 = &tmp[0];
        pol1 = &tmp[nchan];
        for (i=0; i<nchan/2; i++) {
            /* Each loop handles 2 values from each pol */
            memcpy(pol0, in, 2*sizeof(char));
            memcpy(pol1, &in[2], 2*sizeof(char));
            pol0 += 2;
            pol1 += 2;
            in += 4;
        }
    } else if (npol==4) {
        pol0 = &tmp[0];
        pol1 = &tmp[nchan];
        pol2 = &tmp[2*nchan];
        pol3 = &tmp[3*nchan];
        for (i=0; i<nchan; i++) {
            /* Each loop handles one sample */
            *pol0 = *in; in++; pol0++;
            *pol1 = *in; in++; pol1++;
            *pol2 = *in; in++; pol2++;
            *pol3 = *in; in++; pol3++;
        }
    }
    memcpy(b->data + sizeof(long long), tmp, sizeof(char) * npol * nchan);
}

int guppi_udp_close(struct guppi_udp_params *p) {
    close(p->sock);
    return(GUPPI_OK);
}
