/* vegas_params.h 
 *
 * Defines structure used internally to represent observation 
 * parameters.  Includes routines to read/write this info to
 * a "FITS-style" shared memory buffer.
 */
#ifndef _VEGAS_PARAMS_H
#define _VEGAS_PARAMS_H

#include "guppi_udp.h"
#include "sdfits.h"


/* guppi_defines.h
 *
 * Global defines for the upgraded GBT spectrometer software
 */
#define _GUPPI_DEFINES_H

// Increment version number whenever a new version is released
#define SWVER       "1.0"

// Defining NEW_GBT enables the code for the upgraded GBT spectrometer system
#define NEW_GBT         1

// Defining SPEAD enables the decoding of SPEAD packets
#define SPEAD       1

// Defining FAKE_NET enables the generation of fake network data
// and hence disables the network portion of GUPPI.
// #define FAKE_NET    1

// Types of FITS files
#define PSRFITS     1
#define SDFITS      2

// Choose the required type of FITS files
#define FITS_TYPE       SDFITS

// Set SPEAD packet payload size (can be set via cmd line)
#ifndef PAYLOAD_SIZE
    #define PAYLOAD_SIZE    8192
#endif





struct vegas_params {


    /* Packet information for the current block */
    int num_pkts_rcvd;          // Number of packets received in current block
    int num_pkts_dropped;       // Number of packets dropped in current block
    double drop_frac;           // Fraction of dropped packets in this block
    double drop_frac_avg;       // Running average of the fract of dropped packets
    double drop_frac_tot;       // Total fraction of dropped packets

    int stt_valid;              // Has an accurate start time been measured


};




void vegas_read_subint_params(char *buf, 
                              struct vegas_params *g, 
                              struct sdfits *p);
void vegas_read_obs_params(char *buf, 
                           struct vegas_params *g, 
                           struct sdfits *p);
void vegas_free_sdfits(struct sdfits *p);
#endif
