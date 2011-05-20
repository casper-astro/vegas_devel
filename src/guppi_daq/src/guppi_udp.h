/* guppi_udp.h
 *
 * Functions dealing with setting up and 
 * receiving data through a UDP connection.
 */
#ifndef _GUPPI_UDP_H
#define _GUPPI_UDP_H

#include <sys/types.h>
#include <netdb.h>
#include <poll.h>

#define GUPPI_MAX_PACKET_SIZE 9600

/* Struct to hold connection parameters */
struct guppi_udp_params {

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
struct guppi_udp_packet {
    size_t packet_size;  /* packet size, bytes */
    char data[GUPPI_MAX_PACKET_SIZE] __attribute__ ((aligned(32))); /* packet data */
};
unsigned long long guppi_udp_packet_seq_num(const struct guppi_udp_packet *p);
char *guppi_udp_packet_data(const struct guppi_udp_packet *p);
size_t guppi_udp_packet_datasize(size_t packet_size);
size_t parkes_udp_packet_datasize(size_t packet_size);
unsigned long long guppi_udp_packet_flags(const struct guppi_udp_packet *p);

/* Use sender and port fields in param struct to init
 * the other values, bind socket, etc.
 */
int guppi_udp_init(struct guppi_udp_params *p);

/* Wait for available data on the UDP socket */
int guppi_udp_wait(struct guppi_udp_params *p); 

/* Read a packet */
int guppi_udp_recv(struct guppi_udp_params *p, struct guppi_udp_packet *b);

/* Convert a Parkes-style packet to a GUPPI-style packet */
void parkes_to_guppi(struct guppi_udp_packet *b, const int acc_len, 
        const int npol, const int nchan);

/* Copy a guppi packet to the specified location in memory, 
 * expanding out missing channels for 1SFA packets.
 */
void guppi_udp_packet_data_copy(char *out, const struct guppi_udp_packet *p);

/* Copy and corner turn for baseband multichannel modes */
void guppi_udp_packet_data_copy_transpose(char *databuf, int nchan,
        unsigned block_pkt_idx, unsigned packets_per_block,
        const struct guppi_udp_packet *p);

/* Check that the size of the received SPEAD packet is correct */
int guppi_chk_spead_pkt_size(const struct guppi_udp_packet *p);

/* Close out socket, etc */
int guppi_udp_close(struct guppi_udp_params *p);

#endif
