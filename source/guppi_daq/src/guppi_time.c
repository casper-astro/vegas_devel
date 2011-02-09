/* guppi_time.c
 *
 * Routines dealing with time conversion.
 */
#include <stdio.h>
#include <time.h>
#include <sys/time.h>
#include "slalib.h"
#include "guppi_error.h"

int get_current_mjd(int *stt_imjd, int *stt_smjd, double *stt_offs) {
    int rv;
    struct timeval tv;
    struct tm gmt;
    double mjd;

    rv = gettimeofday(&tv,NULL);
    if (rv) { return(GUPPI_ERR_SYS); }

    if (gmtime_r(&tv.tv_sec, &gmt)==NULL) { return(GUPPI_ERR_SYS); }

    slaCaldj(gmt.tm_year+1900, gmt.tm_mon+1, gmt.tm_mday, &mjd, &rv);
    if (rv!=0) { return(GUPPI_ERR_GEN); }

    if (stt_imjd!=NULL) { *stt_imjd = (int)mjd; }
    if (stt_smjd!=NULL) { *stt_smjd = gmt.tm_hour*3600 + gmt.tm_min*60 
        + gmt.tm_sec; }
    if (stt_offs!=NULL) { *stt_offs = tv.tv_usec*1e-6; }

    return(GUPPI_OK);
}

int datetime_from_mjd(long double MJD, int *YYYY, int *MM, int *DD, 
                      int *h, int *m, double *s) {
    int err;
    double fracday;
    
    slaDjcl(MJD, YYYY, MM, DD, &fracday, &err);
    if (err == -1) { return(GUPPI_ERR_GEN); }
    fracday *= 24.0;  // hours
    *h = (int) (fracday);
    fracday = (fracday - *h) * 60.0;  // min
    *m = (int) (fracday);
    *s = (fracday - *m) * 60.0;  // sec
    return(GUPPI_OK);
}

int get_current_lst(double mjd, int *lst_secs) {
    int N = 0;
    double gmst, eqeqx, tt;
    double lon, lat, hgt, lst_rad;
    char scope[10]={"GBT"};
    char name[40];

    // Get Telescope information (currently hardcoded for GBT)
    slaObs(N, scope, name, &lon, &lat, &hgt);
    if (fabs(hgt-880.0) > 0.01) {
        printf("Warning!:  SLALIB is not correctly identifying the GBT!\n\n");
    }
    // These calculations use west longitude is negative
    lon = -lon;

    // Calculate sidereal time of Greenwich (in radians)
    gmst = slaGmst(mjd);

    // Equation of the equinoxes (requires TT)
    tt = mjd + (slaDtt(mjd) / 86400.0);
    eqeqx = slaEqeqx(tt);

    // Local sidereal time = GMST + EQEQX + Longitude in radians
    lst_rad = slaDranrm(gmst + eqeqx + lon);

    // Convert to seconds
    *lst_secs = (int) (lst_rad * 86400.0 / 6.283185307179586476925);

    return(GUPPI_OK);
}

