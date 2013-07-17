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
#include "vegas_params.h"
#include "vegas_time.h"
#include "vegas_error.h"
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
            vegas_error("vegas_read_obs_params", errmsg);               \
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


// Read observation params from a VEGAS machinefits file
void read_machine_params(struct iffits *ifinfo, struct vegas_params *g, struct sdfits *sf)
{
    int temp_int;
    double temp_dbl;

	int nhdu, hdutype, nkeys;
  	long nrows;
	int status=0;

	char *buf;
	char *hdr;

	fits_open_file(&sf->fptr, sf->filename, READONLY, &status);
	fprintf(stderr, "%d\n", status);
	/* Move to correct HDU - don't assume anything about EXTVERs */
	fits_get_num_hdus(sf->fptr, &nhdu, &status);
	fits_movabs_hdu(sf->fptr, 1, &hdutype, &status);
		
	/* get header into the hdr buffer */
	if( fits_hdr2str(sf->fptr, 0, NULL, 0, &hdr, &nkeys, &status ) )
    printf(" Error getting first header\n");
	
	buf = hdr;

    /* Header information */

    /* machine FITS reports NRAO_GBT */
    //get_str("TELESCOP", sf->hdr.telescope, 16, "GBT");
    
    sprintf(sf->hdr.telescope, "GBT");
    get_str("INSTRUME", sf->hdr.instrument, 16, "VEGAS");
    get_str("DATE-OBS", sf->hdr.date_obs, 16, "Unknown");
    get_int("SCAN", temp_int, 1);
    sf->hdr.scan = (double)(temp_int);
    get_int("NCHAN", sf->hdr.nchan, 1024);
    get_str("CAL_MODE", sf->hdr.cal_mode, 16, "Unknown");

	free(hdr);

    printf(" got header\n");


	fits_movabs_hdu(sf->fptr, 4, &hdutype, &status);
	
	
	/* we'll only read first row - assume no doppler tracking */
	fits_read_col(sf->fptr, TDOUBLE, 7, 1, 1, 1, NULL, &temp_dbl, NULL, &status);            
	ifinfo->crval1 = temp_dbl;

	fits_read_col(sf->fptr, TDOUBLE, 8, 1, 1, 1, NULL, &temp_dbl, NULL, &status);            
	ifinfo->cdelt1 = temp_dbl;
	
	printf(" got header\n");

	fits_movabs_hdu(sf->fptr, 6, &hdutype, &status);
	fits_get_num_rows(sf->fptr, &nrows, &status);

	if( fits_hdr2str(sf->fptr, 0, NULL, 0, &hdr, &nkeys, &status ) )
    printf(" Error getting second header\n");
	
	buf=hdr;
    printf(" got header\n");

	
    get_dbl("UTDSTART", sf->hdr.sttmjd, 0);

	get_dbl("UTCSTART", temp_dbl, 0);


    printf(" got utc\n");

    /* Start day and time */

    int YYYY, MM, DD, h, m;
    double s;
	
	sf->hdr.sttmjd = sf->hdr.sttmjd + (temp_dbl/86400.0);

	if(sf->hdr.sttmjd < 1) sf->hdr.sttmjd = 56000.5;
    
	printf("%f\n", sf->hdr.sttmjd);
    
    datetime_from_mjd(sf->hdr.sttmjd, &YYYY, &MM, &DD, &h, &m, &s);
    
    sprintf(sf->hdr.date_obs, "%04d-%02d-%02dT%02d:%02d:%06.3f", 
                YYYY, MM, DD, h, m, s);

	printf("%s\n", sf->hdr.date_obs);
	
	
	fits_close_file(sf->fptr, &status);
	free(hdr);
	
	/* getting the center frequency is turned out to be pretty tough, so we have to jump through some hoops here: */
	
	/* #1: crval1 is being mis-set in the machine fits files, it should be 1/2 of 1 channel */
	ifinfo->crval1 = 1440000000.0/2048.0;
	

	
	
    sf->hdr.freqres = ifinfo->cdelt1;
    sf->hdr.chan_bw = sf->hdr.freqres;
	fprintf(stderr, "chan bw: %f\n", ifinfo->cdelt1);
	sf->hdr.bandwidth = sf->hdr.freqres * sf->hdr.nchan;
	/* for now we'll hard code to AABB */
    sf->hdr.npol = 2;

	/* center of the band */
    sf->hdr.obsfreq = (ifinfo->sff_sideband[ifinfo->currentbank] * (ifinfo->crval1 + ((double) sf->hdr.nchan/2 + 0.5) * ifinfo->cdelt1)) + (ifinfo->sff_multiplier[ifinfo->currentbank] * ifinfo->lo) + ifinfo->sff_offset[ifinfo->currentbank];

	/* #2: the frequency resolution for the LO and sff_offset values doesn't have sufficient resolution to stitch */
	/* our bands together, so we need to demand that the center frequency we derive is an integer number of half-channels */
	/* from our desired band bottom */
	double bandbottom = 10500000000.0;  //for now we want 10.5 GHz
	
	sf->hdr.obsfreq = ((round((sf->hdr.obsfreq - bandbottom) / (sf->hdr.chan_bw/2)) * (sf->hdr.chan_bw/2)) + bandbottom);

	fprintf(stderr, "obsfreq: %f\n", sf->hdr.obsfreq);

    if (status) {
        fprintf(stderr, "Error reading machinefits params.\n");
        fits_report_error(stderr, status);
        exit(1);
    }	

		
	//f_sky = SFF_SIDEBAND * IF + +SFF_MULTIPLIER*LO1 + SFF_OFFSET
	//For a given SUBBAND, IF = CRVAL1 + i * CDELT1 where i goes from 0 to
	//(number of channels-1).

	
}



void read_if_params(struct iffits *ifinfo)
{
    double temp_dbl;


	int nhdu, hdutype, nkeys;
  	long nrows;
	int status=0;
	int i,j,k;

	char *backend[2];
	char *bank[2];
	char *hdr;
	
	for (i = 0; i < 2; i++) {   /* allocate space for string column value */
		backend[i] = (char *) malloc(16);
		bank[i] = (char *) malloc(16);
	}
	ifinfo->N = 0;	
	fits_open_file(&ifinfo->fptr, ifinfo->filename, READONLY, &status);
	fprintf(stderr, "%d\n", status);
	/* Move to correct HDU - don't assume anything about EXTVERs */
	fits_get_num_hdus(ifinfo->fptr, &nhdu, &status);

	fits_movabs_hdu(ifinfo->fptr, 1, &hdutype, &status);

	/* get header into the hdr buffer */
	if( fits_hdr2str(ifinfo->fptr, 0, NULL, 0, &hdr, &nkeys, &status ) )
    printf(" Error getting IF header\n");

	fits_movabs_hdu(ifinfo->fptr, 2, &hdutype, &status);
	fits_get_num_rows(ifinfo->fptr, &nrows, &status);
	for(i=1;i<=nrows;i++) {
	    fits_read_col(ifinfo->fptr, TSTRING, 1, i, 1, 1, NULL, backend, NULL, &status);    
		fits_read_col(ifinfo->fptr, TSTRING, 2, i, 1, 1, NULL, bank, NULL, &status);    


		//fprintf(stderr, "%s %s %s %f\n", backend[0], bank[0], ifinfo->bank[ifinfo->N], temp_dbl);

		for(j = 0; j < ifinfo->N; j++) {
			 if (strcmp(bank[0], ifinfo->bank[j]) == 0) {
				sprintf(bank[0], "Z");	
			 }
		}

		/* if we have a unique vegas band */
		if((strcmp("VEGAS", backend[0]) == 0) && ( strcmp("Z", bank[0]) != 0)){			
			strcpy(ifinfo->bank[ifinfo->N], bank[0]);

		    fits_read_col(ifinfo->fptr, TDOUBLE, 21, i, 1, 1, NULL, &temp_dbl, NULL, &status);            
		    ifinfo->sff_multiplier[ifinfo->N] = temp_dbl;

		    fits_read_col(ifinfo->fptr, TDOUBLE, 22, i, 1, 1, NULL, &temp_dbl, NULL, &status);
		    ifinfo->sff_sideband[ifinfo->N] = temp_dbl;
		    
		    fits_read_col(ifinfo->fptr, TDOUBLE, 23, i, 1, 1, NULL, &temp_dbl, NULL, &status);            
		    ifinfo->sff_offset[ifinfo->N] = temp_dbl;
	
			ifinfo->N = ifinfo->N + 1;
		}

	}

	//fprintf(stderr, "nrows: %ld nkeys: %d nunique %d\n", nrows, nkeys, ifinfo->N);
	fits_close_file(ifinfo->fptr, &status);
	fprintf(stderr, "%d\n", status);
	free(hdr);
}


void read_lo_params(struct iffits *ifinfo)
{
    double temp_dbl;


	int nhdu, hdutype, nkeys;
  	long nrows;
	int status = 0;
	char *hdr;
	
	fits_open_file(&ifinfo->fptr, ifinfo->filename, READONLY, &status);
	fprintf(stderr, "%s %d\n", ifinfo->filename, status);

	
	/* Move to correct HDU - don't assume anything about EXTVERs */
	fits_get_num_hdus(ifinfo->fptr, &nhdu, &status);

	fits_movabs_hdu(ifinfo->fptr, 1, &hdutype, &status);

	/* get header into the hdr buffer */
	if( fits_hdr2str(ifinfo->fptr, 0, NULL, 0, &hdr, &nkeys, &status ) )
    printf(" Error getting LO header\n");

	fits_movabs_hdu(ifinfo->fptr, 4, &hdutype, &status);
	fits_get_num_rows(ifinfo->fptr, &nrows, &status);

	/* we'll only read first row - assume no doppler tracking */
	fits_read_col(ifinfo->fptr, TDOUBLE, 4, 1, 1, 1, NULL, &temp_dbl, NULL, &status);            
	
	ifinfo->lo = temp_dbl;
	
	//fprintf(stderr, "nrows: %ld nkeys: %d nunique %d\n", nrows, nkeys, ifinfo->N);
	fits_close_file(ifinfo->fptr, &status);
	free(hdr);
}

void read_go_params(struct iffits *ifinfo, struct vegas_params *g, struct sdfits *sf)
{
    double temp_dbl;

	char *buf;
	char *hdr;
	int nhdu, hdutype, nkeys;
  	long nrows;
	int status=0;

	fits_open_file(&ifinfo->fptr, ifinfo->filename, READONLY, &status);
	fprintf(stderr, "%d\n", status);

	/* Move to correct HDU - don't assume anything about EXTVERs */
	fits_get_num_hdus(ifinfo->fptr, &nhdu, &status);

	fits_movabs_hdu(ifinfo->fptr, 1, &hdutype, &status);
	
	/* get header into the hdr buffer */
	if( fits_hdr2str(ifinfo->fptr, 0, NULL, 0, &hdr, &nkeys, &status ) )
    printf(" Error getting header\n");
	
	//fprintf(stderr, "nrows: %ld nkeys: %d nunique %d\n", nrows, nkeys, ifinfo->N);
	fits_close_file(ifinfo->fptr, &status);
	buf = hdr;
	
	get_dbl("RA", sf->data_columns.ra, 0.0);
    get_dbl("DEC", sf->data_columns.dec, 0.0);
    get_str("OBJECT", sf->data_columns.object, 16, "Unknown");
    get_str("RECEIVER", sf->hdr.frontend, 16, "Unknown");
    sprintf(sf->hdr.projid, "VEGAS"); 

	free(hdr);
}



