/* vegas_params.c
 *
 * Routines to read sdfits files, functions changed to be compatible
 * guppi_daq psrfits write functions
 * Use PSRFITS style keywords as much as possible.
 */
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <ctype.h>
#include "fitshead.h"
//#include "guppi_params.h"
#include "vegas_params.h"
#include "guppi_time.h"
#include "guppi_error.h"
#include "slalib.h"
#include "sdfits.h"



#ifndef DEGTORAD
#define DEGTORAD 0.017453292519943295769236907684886127134428718885417
#endif
#ifndef RADTODEG
#define RADTODEG 57.29577951308232087679815481410517033240547246656
#endif
#ifndef SOL
#define SOL 299792458.0
#endif

#define DEBUGOUT 0

#define get_dbl(key, param, def) {                                      \
        if (hgetr8(buf, (key), &(param))==0) {                          \
            if (DEBUGOUT)                                               \
                printf("Warning:  %s not in status shm!\n", (key));     \
            (param) = (def);                                            \
        }                                                               \
    }


#define get_flt(key, param, def) {                                      \
        if (hgetr4(buf, (key), &(param))==0) {                          \
            if (DEBUGOUT)                                               \
                printf("Warning:  %s not in status shm!\n", (key));     \
            (param) = (def);                                            \
        }                                                               \
    }



#define get_int(key, param, def) {                                      \
        if (hgeti4(buf, (key), &(param))==0) {                          \
            if (DEBUGOUT)                                               \
                printf("Warning:  %s not in status shm!\n", (key));     \
            (param) = (def);                                            \
        }                                                               \
    }

#define get_lon(key, param, def) {                                      \
        {                                                               \
            double dtmp;                                                \
            if (hgetr8(buf, (key), &dtmp)==0) {                         \
                if (DEBUGOUT)                                           \
                    printf("Warning:  %s not in status shm!\n", (key)); \
                (param) = (def);                                        \
            } else {                                                    \
                (param) = (long long)(rint(dtmp));                      \
            }                                                           \
        }                                                               \
    }

#define get_str(key, param, len, def) {                                 \
        if (hgets(buf, (key), (len), (param))==0) {                     \
            if (DEBUGOUT)                                               \
                printf("Warning:  %s not in status shm!\n", (key));     \
            strcpy((param), (def));                                     \
        }                                                               \
    }

#define exit_on_missing(key, param, val) {                              \
        if ((param)==(val)) {                                           \
            char errmsg[100];                                           \
            sprintf(errmsg, "%s is required!\n", (key));                \
            guppi_error("guppi_read_obs_params", errmsg);               \
            exit(1);                                                    \
        }                                                               \
    }






// Read a status buffer all of the key observation paramters
void vegas_read_subint_params(char *buf, 
                              struct vegas_params *g, 
                              struct sdfits *sf)
{
    int i;
    char subxfreq_str[16];

    // Parse packet size, # of packets, etc.
    get_int("NPKT", g->num_pkts_rcvd, 0);
    get_int("NDROP", g->num_pkts_dropped, 0);
    get_dbl("DROPAVG", g->drop_frac_avg, 0.0);
    get_dbl("DROPTOT", g->drop_frac_tot, 0.0);
    //g->num_heaps = 0;

    if (g->num_pkts_rcvd > 0)
        g->drop_frac = (double) g->num_pkts_dropped / (double) g->num_pkts_rcvd;
    else
        g->drop_frac = 0.0;

    // Valid obs start time
    get_int("STTVALID", g->stt_valid, 0);

    // Observation params
    get_flt("EXPOSURE", sf->data_columns.exposure, 1.0);
    get_str("OBJECT", sf->data_columns.object, 16, "Unknown");
    get_flt("AZ", sf->data_columns.azimuth, 0.0);
    if (sf->data_columns.azimuth < 0.0) sf->data_columns.azimuth += 360.0;
    get_flt("ELEV", sf->data_columns.elevation, 0.0);
    get_flt("BMAJ", sf->data_columns.bmaj, 0.0);
    get_flt("BMIN", sf->data_columns.bmin, 0.0);
    get_flt("BPA", sf->data_columns.bpa, 0.0);
    get_dbl("RA", sf->data_columns.ra, 0.0);
    get_dbl("DEC", sf->data_columns.dec, 0.0);

    // Frequency axis parameters
    sf->data_columns.centre_freq_idx = sf->hdr.nchan/2;

    for(i = 0; i < sf->hdr.nsubband; i++)
    {
        sprintf(subxfreq_str, "SUB%dFREQ", i);
        get_dbl(subxfreq_str, sf->data_columns.centre_freq[i], 0.0);    
    }

    { // MJD and LST calcs
        int imjd, smjd, lst_secs;
        double offs, mjd;
        get_current_mjd(&imjd, &smjd, &offs);
        mjd = (double) imjd + ((double) smjd + offs) / 86400.0;
        get_current_lst(mjd, &lst_secs);
        sf->data_columns.time = (double) lst_secs;
    }

}



// Read a status buffer all of the key observation paramters
void vegas_read_obs_params(char *buf, 
                           struct vegas_params *g, 
                           struct sdfits *sf)
{
    char base[200], dir[200];
    double temp_double;
    int temp_int;

    /* Header information */

    get_str("TELESCOP", sf->hdr.telescope, 16, "GBT");
    get_dbl("BANDWID", sf->hdr.bandwidth, 1e9);
    exit_on_missing("BANDWID", sf->hdr.bandwidth, 0.0);
    get_dbl("NCHAN", temp_double, 2048);
    if(temp_double) sf->hdr.freqres = sf->hdr.bandwidth/temp_double;
    get_dbl("TSYS", sf->hdr.tsys, 0.0);

    get_str("PROJID", sf->hdr.projid, 16, "Unknown");
    get_str("FRONTEND", sf->hdr.frontend, 16, "Unknown");
    get_dbl("OBSFREQ", sf->hdr.obsfreq, 0.0);
    get_int("SCANNUM", temp_int, 1);
    sf->hdr.scan = (double)(temp_int);

    get_str("INSTRUME", sf->hdr.instrument, 16, "VEGAS");
    get_str("CAL_MODE", sf->hdr.cal_mode, 16, "Unknown");
    if (!(strcmp(sf->hdr.cal_mode, "OFF")==0))
    {
        get_dbl("CAL_FREQ", sf->hdr.cal_freq, 25.0);
        get_dbl("CAL_DCYC", sf->hdr.cal_dcyc, 0.5);
        get_dbl("CAL_PHS", sf->hdr.cal_phs, 0.0);
    }
    get_int("NPOL", sf->hdr.npol, 2);
    get_int("NCHAN", sf->hdr.nchan, 1024);
    get_dbl("CHAN_BW", sf->hdr.chan_bw, 1e6);

    get_int("NAXIS1", sf->hdr.nwidth, 16184);
    get_int("NAXIS2", sf->hdr.nrows, 16184);

    get_int("NSUBBAND", sf->hdr.nsubband, 1);
    get_dbl("EFSAMPFR", sf->hdr.efsampfr, 3e9);
    get_dbl("FPGACLK", sf->hdr.fpgaclk, 325e6);
    get_dbl("HWEXPOSR", sf->hdr.hwexposr, 0.5e-3);
    get_dbl("FILTNEP", sf->hdr.filtnep, 0);


    get_str("DATADIR", dir, 200, ".");
    //get_int("FILENUM", sf->filenum, 0);

    /* Start day and time */

    int YYYY, MM, DD, h, m;
    double s;

    get_dbl("STTMJD", sf->hdr.sttmjd, 0.0);
	if(sf->hdr.sttmjd < 1) sf->hdr.sttmjd = 56000.5;
    datetime_from_mjd(sf->hdr.sttmjd, &YYYY, &MM, &DD, &h, &m, &s);
    sprintf(sf->hdr.date_obs, "%04d-%02d-%02dT%02d:%02d:%06.3f", 
                YYYY, MM, DD, h, m, s);
    /* Set the base filename */

    int i;
    char instrument[24];
    strncpy(instrument, sf->hdr.instrument, 24);
    for (i=0; i<24; i++) { 
        if (instrument[i]=='\0') break;
        instrument[i] = tolower(instrument[i]); 
    }
    sprintf(base, "%s_%02d%02d%02d_%04.2f", instrument, DD, MM, YYYY%1000, sf->hdr.scan);
    sprintf(sf->basefilename, "%s/%s", dir, base);

    // We do not set telescope-specific settings
    /* set_obs_params_gb(buf, g, p); */
    
    // Now bookkeeping information
    {
        int bytes_per_dt = sf->hdr.nsubband * sf->hdr.nchan * 4 * 4;
        long long max_bytes_per_file;

        max_bytes_per_file = SDFITS_MAXFILELEN * 1073741824L;

        sf->rows_per_file = max_bytes_per_file / (96 + bytes_per_dt); 

        // Free the old arrays in case we've changed the params
        vegas_free_sdfits(sf);
    }
    
    // Read information that is appropriate for the subints
    vegas_read_subint_params(buf, g, sf);
}



void vegas_free_sdfits(struct sdfits *sd) {
    // Free any memory allocated in to the psrfits struct
}


