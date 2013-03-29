/* vegas_time.h
 *
 * Routines to deal with time convsrsions, etc.
 */
#ifndef _VEGAS_TIME_H
#define _VEGAS_TIME_H

#include "vegas_defines.h"

/* Return current time using PSRFITS-style integer MJD, integer 
 * second time of day, and fractional second offset. */
int get_current_mjd(int *stt_imjd, int *stt_smjd, double *stt_offs);

/* Return Y, M, D, h, m, and s for an MJD */
int datetime_from_mjd(long double MJD, int *YYYY, int *MM, int *DD, 
                      int *h, int *m, double *s);

/* Return the LST (in sec) for the GBT at a specific MJD (UTC) */
int get_current_lst(double mjd, int *lst_secs);

#ifdef NEW_GBT

/* Return the time in MJD, with microsecond resolution */
int get_current_mjd_double(double *mjd);

#endif

#endif
