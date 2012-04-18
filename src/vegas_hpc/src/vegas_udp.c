/* vegas_udp.c
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
#include <errno.h>

#include "vegas_udp.h"
#include "vegas_databuf.h"
#include "vegas_error.h"
#include "vegas_defines.h"

#ifdef NEW_GBT
#define BYTE_ARR_TO_UINT(array, idx) (ntohl(((unsigned int*)(array))[idx]))
#endif

int vegas_udp_init(struct vegas_udp_params *p) {

    /* Resolve sender hostname */
    struct addrinfo hints;
    struct addrinfo *result, *rp;
    memset(&hints, 0, sizeof(struct addrinfo));
    hints.ai_family = AF_INET;
    hints.ai_socktype = SOCK_DGRAM;
    int rv = getaddrinfo(p->sender, NULL, &hints, &result);
    if (rv!=0) { 
        vegas_error("vegas_udp_init", "getaddrinfo failed");
        return(VEGAS_ERR_SYS);
    }

    /* Set up socket */
    p->sock = socket(PF_INET, SOCK_DGRAM, 0);
    if (p->sock==-1) { 
        vegas_error("vegas_udp_init", "socket error");
        freeaddrinfo(result);
        return(VEGAS_ERR_SYS);
    }

    /* bind to local address */
    struct sockaddr_in local_ip;
    local_ip.sin_family =  AF_INET;
    local_ip.sin_port = htons(p->port);
    local_ip.sin_addr.s_addr = INADDR_ANY;
    rv = bind(p->sock, (struct sockaddr *)&local_ip, sizeof(local_ip));
    if (rv==-1) {
        vegas_error("vegas_udp_init", "bind");
        return(VEGAS_ERR_SYS);
    }

    /* Set up socket to recv only from sender */
    for (rp=result; rp!=NULL; rp=rp->ai_next) {
        if (connect(p->sock, rp->ai_addr, rp->ai_addrlen)==0) { break; }
    }
    if (rp==NULL) { 
        vegas_error("vegas_udp_init", "connect error");
        close(p->sock); 
        freeaddrinfo(result);
        return(VEGAS_ERR_SYS);
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
        vegas_error("vegas_udp_init", "Error setting rcvbuf size.");
        perror("setsockopt");
    } 
    rv = getsockopt(p->sock, SOL_SOCKET, SO_RCVBUF, &bufsize, &ss); 
    if (0 && rv==0) { 
        printf("vegas_udp_init: SO_RCVBUF=%d\n", bufsize);
    }

    /* Poll command */
    p->pfd.fd = p->sock;
    p->pfd.events = POLLIN;

    return(VEGAS_OK);
}

int vegas_udp_wait(struct vegas_udp_params *p) {
    int rv = poll(&p->pfd, 1, 1000); /* Timeout 1sec */
    if (rv==1) { return(VEGAS_OK); } /* Data ready */
    else if (rv==0) { return(VEGAS_TIMEOUT); } /* Timed out */
    else { 
        /* EINTR is not actually an error */
        if (errno == EINTR) {
            printf("Got interrupted system call (EINTR)... continuing\n");
            return(VEGAS_TIMEOUT);
        }
        return(VEGAS_ERR_SYS); /* Other error */
    }  
}

int vegas_udp_recv(struct vegas_udp_params *p, struct vegas_udp_packet *b) {
    int rv = recv(p->sock, b->data, VEGAS_MAX_PACKET_SIZE, 0);
    b->packet_size = rv;
    if (rv==-1) { return(VEGAS_ERR_SYS); }
    else if (p->packet_size) {
#ifdef SPEAD

    	if (strncmp(p->packet_format, "SPEAD", 5) == 0)
            return vegas_chk_spead_pkt_size(b);
		else if (rv!=p->packet_size)
			return(VEGAS_ERR_PACKET);
		else
            return(VEGAS_OK);
#else
        if (rv!=p->packet_size) { return(VEGAS_ERR_PACKET); }
        else { return(VEGAS_OK); }
#endif
    } else { 
        p->packet_size = rv;
        return(VEGAS_OK); 
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

unsigned long long vegas_udp_packet_seq_num(const struct vegas_udp_packet *p) {
    // XXX Temp for new baseband mode, blank out top 8 bits which 
    // contain channel info.

    unsigned long long tmp = change_endian64((unsigned long long *)p->data);
    tmp &= 0x00FFFFFFFFFFFFFF;
    return(tmp);
    //return(change_endian64((unsigned long long *)(p->data)));
}


#define PACKET_SIZE_ORIG ((size_t)8208)
#define PACKET_SIZE_SHORT ((size_t)544)
#define PACKET_SIZE_1SFA ((size_t)8224)
#define PACKET_SIZE_1SFA_OLD ((size_t)8160)
#define PACKET_SIZE_FAST4K ((size_t)4128)
#define PACKET_SIZE_PASP ((size_t)528)
#define PACKET_SIZE_SPEAD ((size_t)8248)

size_t vegas_udp_packet_datasize(size_t packet_size) {
    /* Special case for the new "1SFA" packets, which have an extra
     * 16 bytes at the end reserved for future use.  All other vegas
     * packets have 8 bytes index at the front, and 8 bytes error
     * flags at the end.
     * NOTE: This represents the "full" packet output size...
     */
    if (packet_size==PACKET_SIZE_1SFA) // 1SFA packet size
        return((size_t)8192);
    else if (packet_size==PACKET_SIZE_SHORT) 
        //return((size_t)256);
        return((size_t)512);
	else if (packet_size==PACKET_SIZE_SPEAD)
		return(packet_size - 6*8); // 8248-6*8
    else              
        return(packet_size - 2*sizeof(unsigned long long));
}

char *vegas_udp_packet_data(const struct vegas_udp_packet *p) {
    /* This is valid for all vegas packet formats
     * PASP has 16 bytes of header rather than 8.
     */
    if (p->packet_size==PACKET_SIZE_PASP)
        return((char *)(p->data) + (size_t)16);
    return((char *)(p->data) + sizeof(unsigned long long));
}

unsigned long long vegas_udp_packet_flags(const struct vegas_udp_packet *p) {
    return(*(unsigned long long *)((char *)(p->data) 
                + p->packet_size - sizeof(unsigned long long)));
}

/* Copy the data portion of a vegas udp packet to the given output
 * address.  This function takes care of expanding out the 
 * "missing" channels in 1SFA packets.
 */
void vegas_udp_packet_data_copy(char *out, const struct vegas_udp_packet *p) {
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
        memcpy(out + pad, vegas_udp_packet_data(p), spec_data_size);
        memset(out + pad + spec_data_size, 0, 2*pad);
        memcpy(out + pad + spec_data_size + pad + pad, 
                vegas_udp_packet_data(p) + spec_data_size, 
                spec_data_size);
        memset(out + pad + spec_data_size + pad
                + pad + spec_data_size, 0, pad);
    } else {
        /* Packet has full data, just do a memcpy */
        memcpy(out, vegas_udp_packet_data(p), 
                vegas_udp_packet_datasize(p->packet_size));
    }
}

/* Copy function for baseband data that does a partial
 * corner turn (or transpose) based on nchan.  In this case
 * out should point to the beginning of the data buffer.
 * block_pkt_idx is the seq number of this packet relative
 * to the beginning of the block.  packets_per_block
 * is the total number of packets per data block (all channels).
 */
void vegas_udp_packet_data_copy_transpose(char *databuf, int nchan,
        unsigned block_pkt_idx, unsigned packets_per_block,
        const struct vegas_udp_packet *p) {
    const unsigned chan_per_packet = nchan;
    const size_t bytes_per_sample = 4;
    const unsigned samp_per_packet = vegas_udp_packet_datasize(p->packet_size) 
        / bytes_per_sample / chan_per_packet;
    const unsigned samp_per_block = packets_per_block * samp_per_packet;

    char *iptr, *optr;
    unsigned isamp,ichan;
    iptr = vegas_udp_packet_data(p);

    for (isamp=0; isamp<samp_per_packet; isamp++) {
        optr = databuf + bytes_per_sample * (block_pkt_idx*samp_per_packet 
                + isamp);
        for (ichan=0; ichan<chan_per_packet; ichan++) {
            memcpy(optr, iptr, bytes_per_sample);
            iptr += bytes_per_sample;
            optr += bytes_per_sample*samp_per_block;
        }
    }

#if 0 
    // Old version...
    const unsigned pkt_idx = block_pkt_idx / nchan;
    const unsigned ichan = block_pkt_idx % nchan;
    const unsigned offset = ichan * packets_per_block / nchan + pkt_idx;
    memcpy(databuf + offset*vegas_udp_packet_datasize(p->packet_size), 
            vegas_udp_packet_data(p),
            vegas_udp_packet_datasize(p->packet_size));
#endif
}

size_t parkes_udp_packet_datasize(size_t packet_size) {
    return(packet_size - sizeof(unsigned long long));
}

void parkes_to_vegas(struct vegas_udp_packet *b, const int acc_len, 
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
    char tmp[VEGAS_MAX_PACKET_SIZE];
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


#ifdef SPEAD

/* Check that the size of the received SPEAD packet is correct.
 * This is acheived by reading the size fields in the SPEAD packet header,
 * and comparing them to the actual size of the received packet. */
int vegas_chk_spead_pkt_size(const struct vegas_udp_packet *p)
{
	unsigned int spead_hdr_upr = 0x53040305;
	int num_items, payload_size;
    int i;

    //Confirm we have enough bytes for header + 3 fields
    if(p->packet_size < 8*4)
        return (VEGAS_ERR_PACKET);
    
    //Check that the header is valid
    if( BYTE_ARR_TO_UINT(p->data, 0) != spead_hdr_upr )
        return (VEGAS_ERR_PACKET);

    //Get number of items (from last 2 bytes of header)
    num_items = p->data[6]<<8 | p->data[7];

    payload_size = -1;

    //Get packet payload length, by searching through the fields
    for(i = 8; i < (8 + num_items*8); i+=8)
    {
        //If we found the packet payload length item
        if( (p->data[i+1]<<8 | p->data[i+2]) == 4 )
        {
            payload_size = BYTE_ARR_TO_UINT(p->data, i/4 + 1);
            break;
        }
    }
    
    if(payload_size == -1)
        return (VEGAS_ERR_PACKET);

    //Confirm that packet size is correct
    if(p->packet_size != 8 + num_items*8 + payload_size)
        return (VEGAS_ERR_PACKET);

    return (VEGAS_OK);
}

unsigned int vegas_spead_packet_heap_cntr(const struct vegas_udp_packet *p)
{
    int i;

    //Get heap counter, by searching through the fields
    for(i = 8; i < (8 + 4*8); i+=8)
    {
        //If we found the heap counter item
        if( (p->data[i+1]<<8 | p->data[i+2]) == 1 )
        {
            return BYTE_ARR_TO_UINT(p->data, i/4 + 1);
        }
    }
    
    return (VEGAS_ERR_PACKET);
}


unsigned int vegas_spead_packet_heap_offset(const struct vegas_udp_packet *p)
{
    int i;

    //Get heap offset, by searching through the fields
    for(i = 8; i < (8 + 4*8); i+=8)
    {
        //If we found the heap offset item
        if( (p->data[i+1]<<8 | p->data[i+2]) == 3 )
        {
            return BYTE_ARR_TO_UINT(p->data, i/4 + 1);
        }
    }
    
    return (VEGAS_ERR_PACKET);
}


unsigned int vegas_spead_packet_seq_num(int heap_cntr, int heap_offset, int packets_per_heap)
{
    return (heap_cntr * packets_per_heap) + (heap_offset / PAYLOAD_SIZE);
}


char* vegas_spead_packet_data(const struct vegas_udp_packet *p)
{
    return (char*)(p->data + 5*8);
}


unsigned int vegas_spead_packet_datasize(const struct vegas_udp_packet *p)
{
    return p->packet_size - 5*8;
}


int vegas_spead_packet_copy(struct vegas_udp_packet *p, char *header_addr,
                            char* payload_addr, char bw_mode[])
{
    int i, num_items;
    char* pkt_payload;
    int payload_size, offset;
    int bw = (strncmp(bw_mode, "high", 4) == 0);

    num_items = p->data[6]<<8 | p->data[7];

    /* Copy header, reversing both the ID and value of each field */
    for(i = 0; i < num_items - 4; i++)
    {
        header_addr[0]
                = vegas_spead_packet_data(p)[i*8];                                  //mode byte
        *((unsigned short *)(header_addr + 1))
                = ntohs(*(unsigned short *)(vegas_spead_packet_data(p) + i*8 + 1)); //ID (16 bits)
        header_addr[3]
                = vegas_spead_packet_data(p)[i*8 + 3];                              //pad byte
        *((unsigned int *)(header_addr + 4))
                = ntohl(*(unsigned int *)(vegas_spead_packet_data(p) + i*8 + 4));   //value (32 bits)
        if (bw && (*((unsigned short *)(header_addr + 1)) == 0x22)) {    // 0x22 is Integration_size
            *((unsigned int *)(header_addr + 4)) += 1;                  // for hbw modes add 1 here so FPGA doesn't have to
        }
//        printf("%d : %d  ",*((unsigned short *)(header_addr + 1)),*((unsigned int *)(header_addr + 4)));
        header_addr = (char*)(header_addr + 8);
    }
//    printf("\n");
    

    /* Copy payload */
    
    pkt_payload = vegas_spead_packet_data(p) + (num_items - 4) * 8;
    payload_size = vegas_spead_packet_datasize(p) - (num_items - 4) * 8;

    /* If high-bandwidth mode */
    if(bw)
    {
        for(offset = 0; offset < payload_size; offset += 4)
        {
            *(unsigned int *)(payload_addr + offset) =
                ntohl(*(unsigned int *)(pkt_payload + offset));
        }
    }
    
    /* Else if low-bandwidth mode */
    else if(strncmp(bw_mode, "low", 3) == 0)
        memcpy(payload_addr, pkt_payload, payload_size);

    return 0;
}

#endif


int vegas_udp_close(struct vegas_udp_params *p) {
    close(p->sock);
    return(VEGAS_OK);
}
