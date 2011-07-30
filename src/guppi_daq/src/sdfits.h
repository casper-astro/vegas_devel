/* sdfits.h */
#ifndef _SDFITS_H
#define _SDFITS_H
#include "fitsio.h"

// The following is the max file length in GB
#define SDFITS_MAXFILELEN 1L

// The following is the template file to use to create a PSRFITS file.
// Path is relative to GUPPI_DIR environment variable.
#define SDFITS_TEMPLATE "src/guppi_SDFITS_template.txt"

struct primary_hdrinfo
{
    char date[16];          // Date file was created (dd/mm/yy)
};

struct hdrinfo
{
    char telescope[16];     // Telescope used
    double bandwidth;       // Bandwidth of the entire backend
    double freqres;         // Width of each spectral channel in the file
    char date_obs[16];      // Date of observation (dd/mm/yy)
    double tsys;            // System temperature

    char projid[16];        // The project ID
    char frontend[16];      // Frontend used
    double obsfreq;         // Centre frequency for observation
    double lst;             // LST (seconds after 0h) at start of scan
    double scan;            // Scan number (float)

    char instrument[16];       // Backend or instrument used
    char cal_mode[16];      // Cal mode (OFF, SYNC, EXT1, EXT2
    double cal_freq;        // Cal modulation frequency (Hz)
    double cal_dcyc;        // Cal duty cycle (0-1)
    double cal_phs;         // Cal phase (wrt start time)
    int npol;               // Number of antenna polarisations (normally 2)
    int nchan;              // Number of spectral bins per sub-band
    double chan_bw;         // Width of each spectral bin

    int nsubband;           // Number of sub-bands
};
    
struct sdfits_data_columns
{
    double time;            // UT seconds at start of integration (since 0h UT)
    float exposure;         // Effective integration time (seconds)
    char object[16];        // Object being viewed
    float azimuth;          // Commanded azimuth
    float elevation;        // Commanded elevation
    float bmaj;             // Beam major axis length (deg)
    float bmin;             // Beam minor axis length (deg)
    float bpa;              // Beam position angle (deg)

    int accumid;            // ID of the accumulator from where the spectrum came
    int sttspec;            // SPECTRUM_COUNT of the first spectrum in the integration
    int stpspec;            // SPECTRUM_COUNT of the last spectrum in the integration

    float centre_freq_idx;  // Index of centre frequency bin
    double centre_freq[8];  // Frequency at centre of each sub-band
    double ra;              // RA mid-integration
    double dec;             // DEC mid-integration

    char data_len[16];      // Length of the data array
    char data_dims[16];     // Data matrix dimensions
    unsigned char *data;    // Ptr to the raw data itself
};

struct sdfits
{
    char basefilename[200]; // The base filename from which to build the true filename
    char filename[200];     // Filename of the current PSRFITs file
    long long N;            // Current number of spectra written
    double T;               // Current duration of the observation written
    int filenum;            // The current number of the file in the scan (1-offset)
    int new_file;           // Indicates that a new file must be created.    
    int rownum;             // The current data row number to be written (1-offset)
    int tot_rows;           // The total number of data rows written so far
    int rows_per_file;      // The maximum number of data rows per file
    int status;             // The CFITSIO status value
    fitsfile *fptr;         // The CFITSIO file structure
    int multifile;          // Write multiple output files
    int quiet;              // Be quiet about writing each subint
    char mode;              // Read (r) or write (w).
    struct primary_hdrinfo primary_hdr;
    struct hdrinfo hdr;
    struct sdfits_data_columns data_columns;
};

// In write_sdfits.c
int sdfits_create(struct sdfits *sf);
int sdfits_close(struct sdfits *sf);
int sdfits_write_subint(struct sdfits *sf);

// In read_psrfits.c
// int psrfits_open(struct psrfits *pf);
// int psrfits_read_subint(struct psrfits *pf);
// int psrfits_read_part_DATA(struct psrfits *pf, int N, char *buffer);

#endif
