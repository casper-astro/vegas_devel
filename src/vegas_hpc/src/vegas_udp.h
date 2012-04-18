/* vegas_udp.h
 *
 * Functions dealing with setting up and 
 * receiving data through a UDP connection.
 */
#ifndef _VEGAS_UDP_H
#define _VEGAS_UDP_H

#include <sys/types.h>
#include <netdb.h>
#include <poll.h>
#include "vegas_defines.h"

#define VEGAS_MAX_PACKET_SIZE 9600

/* Struct to hold connection parameters */
struct vegas_udp_params {

    /* Info needed from outside: */
    char sender[80];  /* Sender hostname */
    int port;         /* Receive port */
    size_t packet_size;     /* Expected packet size, 0 = don't care */
    char packet_format[32]; /* Packet format */

    /* Derived from above: */
    int sock;                       /* Receive socket */
    struct addrinfo sender_addr;    /* Sender hostname/IP params */
    struct pollfd pfd;              /* Use to poll for avail data */
};

/* Basic structure of a packet.  This struct, functions should 
 * be used to get the various components of a data packet.   The
 * internal packet structure is:
 *   1. sequence number (64-bit unsigned int)
 *   2. data bytes (typically 8kB)
 *   3. status flags (64 bits)
 *
 * Except in the case of "1SFA" packets:
 *   1. sequence number (64b uint)
 *   2. data bytes (typically 8128B)
 *   3. status flags (64b)
 *   4. blank space (16B)
 */
struct vegas_udp_packet {
    size_t packet_size;  /* packet size, bytes */
    char data[VEGAS_MAX_PACKET_SIZE] __attribute__ ((aligned(32))); /* packet data */
};
unsigned long long vegas_udp_packet_seq_num(const struct vegas_udp_packet *p);
char *vegas_udp_packet_data(const struct vegas_udp_packet *p);
size_t vegas_udp_packet_datasize(size_t packet_size);
size_t parkes_udp_packet_datasize(size_t packet_size);
unsigned long long vegas_udp_packet_flags(const struct vegas_udp_packet *p);

/* Use sender and port fields in param struct to init
 * the other values, bind socket, etc.
 */
int vegas_udp_init(struct vegas_udp_params *p);

/* Wait for available data on the UDP socket */
int vegas_udp_wait(struct vegas_udp_params *p); 

/* Read a packet */
int vegas_udp_recv(struct vegas_udp_params *p, struct vegas_udp_packet *b);

/* Convert a Parkes-style packet to a VEGAS-style packet */
void parkes_to_vegas(struct vegas_udp_packet *b, const int acc_len, 
        const int npol, const int nchan);

/* Copy a vegas packet to the specified location in memory, 
 * expanding out missing channels for 1SFA packets.
 */
void vegas_udp_packet_data_copy(char *out, const struct vegas_udp_packet *p);

/* Copy and corner turn for baseband multichannel modes */
void vegas_udp_packet_data_copy_transpose(char *databuf, int nchan,
        unsigned block_pkt_idx, unsigned packets_per_block,
        const struct vegas_udp_packet *p);

#ifdef SPEAD

/* Check that the size of the received SPEAD packet is correct */
int vegas_chk_spead_pkt_size(const struct vegas_udp_packet *p);

unsigned int vegas_spead_packet_heap_cntr(const struct vegas_udp_packet *p);
unsigned int vegas_spead_packet_heap_offset(const struct vegas_udp_packet *p);
unsigned int vegas_spead_packet_seq_num(int heap_cntr, int heap_offset, int packets_per_heap);
char* vegas_spead_packet_data(const struct vegas_udp_packet *p);
unsigned int vegas_spead_packet_datasize(const struct vegas_udp_packet *p);
int vegas_spead_packet_copy(struct vegas_udp_packet *p, char *header_addr,
                            char *payload_addr, char bw_mode[]);

#endif

/* Close out socket, etc */
int vegas_udp_close(struct vegas_udp_params *p);

#endif
