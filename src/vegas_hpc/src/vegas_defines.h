/* vegas_defines.h
 *
 * Global defines for the upgraded GBT spectrometer software
 */
#ifndef _VEGAS_DEFINES_H
#define _VEGAS_DEFINES_H

// Increment version number whenever a new version is released
#define SWVER       "1.0"

// Defining NEW_GBT enables the code for the upgraded GBT spectrometer system
#define NEW_GBT		1

// Defining SPEAD enables the decoding of SPEAD packets
#define SPEAD       1

// Defining FAKE_NET enables the generation of fake network data
// and hence disables the network portion of VEGAS.
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

// The rate at which data blocks are written to disk (in seconds)
#define DISK_WRITE_INTERVAL 20

#endif
