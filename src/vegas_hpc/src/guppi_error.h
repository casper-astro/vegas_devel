/* guppi_error.h
 *
 * Error handling routines for guppi.
 */
#ifndef _GUPPI_ERROR_H
#define _GUPPI_ERROR_H

/* Some exit codes */
#define GUPPI_OK          0
#define GUPPI_TIMEOUT     1 // Call timed out 
#define GUPPI_ERR_GEN    -1 // Super non-informative
#define GUPPI_ERR_SYS    -2 // Failed system call
#define GUPPI_ERR_PARAM  -3 // Parameter out of range
#define GUPPI_ERR_KEY    -4 // Requested key doesn't exist
#define GUPPI_ERR_PACKET -5 // Unexpected packet size

#define DEBUGOUT 0 

/* Call this to log an error message */
void guppi_error(const char *name, const char *msg);

/* Call this to log an warning message */
void guppi_warn(const char *name, const char *msg);

#endif
