/* psrfits.h */
#ifndef _PSRFITS_H
#define _PSRFITS_H
#include "fitsio.h"

// The following is the max file length in GB, different for fold/search
#define PSRFITS_MAXFILELEN_SEARCH 10L
#define PSRFITS_MAXFILELEN_FOLD 1L

// The following is the template file to use to create a PSRFITS file.
// Path is relative to GUPPI_DIR environment variable.
#define PSRFITS_SEARCH_TEMPLATE "src/guppi_PSRFITS_v3.4_search_template.txt"
#define PSRFITS_FOLD_TEMPLATE "src/guppi_PSRFITS_v3.4_fold_template.txt"

struct hdrinfo {
    char obs_mode[8];       // Observing mode (SEARCH, PSR, CAL)
    char telescope[24];     // Telescope used
    char observer[24];      // Observer's name
    char source[24];        // Source name
    char frontend[24];      // Frontend used
    char backend[24];       // Backend or instrument used
    char project_id[24];    // Project identifier
    char date_obs[24];      // Start of observation (YYYY-MM-DDTHH:MM:SS.SSS)
    char ra_str[16];        // Right Ascension string (HH:MM:SS.SSSS)
    char dec_str[16];       // Declination string (DD:MM:SS.SSSS)
    char poln_type[8];      // Polarization recorded (LIN or CIRC)
    char poln_order[16];    // Order of polarizations (i.e. XXYYXYYX)
    char track_mode[16];    // Track mode (TRACK, SCANGC, SCANLAT)
    char cal_mode[8];       // Cal mode (OFF, SYNC, EXT1, EXT2
    char feed_mode[8];      // Feed track mode (FA, CPA, SPA, TPA)
    long double MJD_epoch;  // Starting epoch in MJD
    double dt;              // Sample duration (s)
    double fctr;            // Center frequency of the observing band (MHz)
    double orig_df;         // Original frequency spacing between the channels (MHz)
    double df;              // Frequency spacing between the channels (MHz)
    double BW;              // Bandwidth of the observing band (MHz)
    double ra2000;          // RA  of observation (deg, J2000) at obs start
    double dec2000;         // Dec of observation (deg, J2000) at obs start
    double azimuth;         // Azimuth (commanded) at the start of the obs (deg)
    double zenith_ang;      // Zenith angle (commanded) at the start of the obs (deg)
    double beam_FWHM;       // Beam FWHM (deg)
    double cal_freq;        // Cal modulation frequency (Hz)
    double cal_dcyc;        // Cal duty cycle (0-1)
    double cal_phs;         // Cal phase (wrt start time)
    double feed_angle;      // Feed/Posn angle requested (deg)
    double scanlen;         // Requested scan length (sec)
    double start_lst;       // Start LST (sec past 00h)
    double start_sec;       // Start time (sec past UTC 00h) 
    double chan_dm;         // DM that each channel was de-dispersed at (pc/cm^3)
    double fd_sang;         // Reference angle for feed rotation (deg)
    double fd_xyph;         // Cal signal poln cross-term phase (deg)
    int start_day;          // Start MJD (UTC days) (J - long integer)
    int scan_number;        // Number of scan
    int nbits;              // Number of bits per data sample 
    int nbin;               // Number of bins per period in fold mode
    int nchan;              // Number of channels
    int npol;               // Number of polarizations to be stored (1 for summed)
    int nsblk;              // Number of spectra per row
    int orig_nchan;         // Number of spectral channels per sample
    int summed_polns;       // Are polarizations summed? (1=Yes, 0=No)
    int rcvr_polns;         // Number of polns provided by the receiver
    int offset_subint;      // Offset subint number for first row in the file
    int ds_time_fact;       // Software downsampling factor in time (1 if none)
    int ds_freq_fact;       // Software downsampling factor in freq (1 if none)
    int onlyI;              // 1 if the software will only record Stokes I
    int fd_hand;            // Receiver "handedness" or X/Y swap (+/-1)
    int be_phase;           // Backend poln cross-term phase convention (+/-1)
};

struct subint {
    double tsubint;         // Length of subintegration (sec)
    double offs;            // Offset from Start of subint centre (sec)
    double lst;             // LST at subint centre (sec)
    double ra;              // RA (J2000) at subint centre (deg)
    double dec;             // Dec (J2000) at subint centre (deg)
    double glon;            // Gal longitude at subint centre (deg)
    double glat;            // Gal latitude at subint centre (deg)
    double feed_ang;        // Feed angle at subint centre (deg)
    double pos_ang;         // Position angle of feed at subint centre (deg)
    double par_ang;         // Parallactic angle at subint centre (deg)
    double tel_az;          // Telescope azimuth at subint centre (deg)
    double tel_zen;         // Telescope zenith angle at subint centre (deg)
    int bytes_per_subint;   // Number of bytes for one row of raw data
    int FITS_typecode;      // FITS data typecode as per CFITSIO
    float *dat_freqs;       // Ptr to array of Centre freqs for each channel (MHz)
    float *dat_weights;     // Ptr to array of Weights for each channel
    float *dat_offsets;     // Ptr to array of offsets for each chan * pol
    float *dat_scales;      // Ptr to array of scalings for each chan * pol
    unsigned char *data;    // Ptr to the raw data itself
};

#include "polyco_struct.h"
struct foldinfo {
    char parfile[256];      // Parfile name for folding
    int n_polyco_sets;      // Number of polyco sets present
    struct polyco *pc;      // Pointer to polyco blocks
    int nbin;               // Requested number of bins
    double tfold;           // Requested fold integration time
};

struct psrfits {
    char basefilename[200]; // The base filename from which to build the true filename
    char filename[200];     // Filename of the current PSRFITs file
    long long N;            // Current number of spectra written
    double T;               // Current duration of the observation written
    int filenum;            // The current number of the file in the scan (1-offset)
    int rownum;             // The current subint number to be written (1-offset)
    int tot_rows;           // The total number of subints written so far
    int rows_per_file;      // The maximum number of rows (subints) per file
    int status;             // The CFITSIO status value
    fitsfile *fptr;         // The CFITSIO file structure
    int multifile;          // Write multiple output files
    int quiet;              // Be quiet about writing each subint
    char mode;              // Read (r) or write (w).
    struct hdrinfo hdr;
    struct subint sub;
    struct foldinfo fold;   
};

// In write_psrfits.c
int psrfits_create(struct psrfits *pf);
int psrfits_write_subint(struct psrfits *pf);
int psrfits_write_polycos(struct psrfits *pf, struct polyco *pc, int npc);
int psrfits_write_ephem(struct psrfits *pf, FILE *parfile);
int psrfits_close(struct psrfits *pf);
#define SEARCH_MODE 1
#define FOLD_MODE 2
int psrfits_obs_mode(const char *obs_mode);
int psrfits_remove_polycos(struct psrfits *pf);
int psrfits_remove_ephem(struct psrfits *pf);

// In read_psrfits.c
int psrfits_open(struct psrfits *pf);
int psrfits_read_subint(struct psrfits *pf);
int psrfits_read_part_DATA(struct psrfits *pf, int N, char *buffer);

#endif
