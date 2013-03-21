#include <stdio.h>
#include <stdlib.h>
#include "fitsio.h"
#include "psrfits.h"
//#include "guppi_params.h"
#include "vegas_params.h"
#include "fitshead.h"
#include "guppi_time.h"

#include <math.h>
#include <arpa/inet.h>
#include <string.h>
#include "median.h"
#include <sys/stat.h>

/* set to 1 to recalculate stats continuously */
#define CONTINUOUS_QUANT 1

/* number of spectra to put in each subint */
#define SPECTRA_PER_ROW 4096

/* total number of subints per file */
#define ROWS_PER_FILE 1000000

/* first channel to take */
#define CHANSTART 72

/*FAKE PULSAR TEST*/
//#define CHANSTART 1

/* last channel to keep */
#define CHANEND 994
//#define CHANEND 1023


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


struct iffits
{
	char filename[200];     	// Filename of the IF fits file
	char observation[100];
	char basefilename[100];
	char *bank[8];				// VEGAS banks used in this obs
	int N;            			// Number of VEGAS banks used
	int currentbank;			//current bank 
	int writtenbank;
	double sff_multiplier[8];
	double sff_sideband[8];
	double sff_offset[8];
	double lo;
	double crval1;
	double cdelt1;
	fitsfile *fptr;         	// The CFITSIO file structure
};  



double round(double x);
unsigned char uquantize(float d, float min, float max);             
int exists(const char *fname);

void read_if_params(struct iffits *ifinfo);
void read_lo_params(struct iffits *ifinfo);
void read_go_params(struct iffits *ifinfo, struct vegas_params *g, struct sdfits *sf);

void read_machine_params(struct iffits *ifinfo, struct vegas_params *g, struct sdfits *p);

int sdfits_to_psrfits_write_subint(struct sdfits *sf, struct psrfits *pf);
int sdfits_to_psrfits_create(struct sdfits *sf, struct psrfits *pf, struct iffits *ifinfo);
void quant(float *data, unsigned char * quantbytes, int nchan, int npol, int nframe, int sigma, float *mads);
void massage(float *data, int nchan, int npol, int nframe, float *medians, float *mads, float *means);
             
void dec2hms(char *out, double in, int sflag) {
    int sign = 1;
    char *ptr = out;
    int h, m;
    double s;
    if (in<0.0) { sign = -1; in = fabs(in); }
    h = (int)in; in -= (double)h; in *= 60.0;
    m = (int)in; in -= (double)m; in *= 60.0;
    s = in;
    if (sign==1 && sflag) { *ptr='+'; ptr++; }
    else if (sign==-1) { *ptr='-'; ptr++; }
    sprintf(ptr, "%2.2d:%2.2d:%07.4f", h, m, s);
}


/* IMSWAP4 -- Reverse bytes of Integer*4 or Real*4 vector in place */
void
imswap4 (string,nbytes)

char *string;   /* Address of Integer*4 or Real*4 vector */
int nbytes;     /* Number of bytes to reverse */

{
    char *sbyte, *slast;
    char temp0, temp1, temp2, temp3;

    slast = string + nbytes;
    sbyte = string;
    while (sbyte < slast) {
        temp3 = sbyte[0];
        temp2 = sbyte[1];
        temp1 = sbyte[2];
        temp0 = sbyte[3];
        sbyte[0] = temp0;
        sbyte[1] = temp1;
        sbyte[2] = temp2;
        sbyte[3] = temp3;
        sbyte = sbyte + 4;
        }

    return;
}

          
              
int main(int argc, char *argv[]) {
	struct vegas_params vf;
    struct sdfits sf;

    struct psrfits pf;
	//struct guppi_params gf;

  

	struct iffits ifinfo;

    char buf[65536];
    char filname[250];
	int filepos=0;
    char *hdr;
    
    FILE *fil;   //input file
    

	memset(buf, 0x0, 65536);
	long int i,j,k;
    

    if(argc < 2) {
		fprintf(stderr, "USAGE: %s PROJECT/IF/observation.fits <optional, data directory>\n", argv[0]);
		exit(1);
	}


    
    
	strcpy(ifinfo.observation, strrchr(argv[1], 47) + 1);
	memcpy(ifinfo.basefilename, argv[1], strlen(argv[1]) - (strlen(ifinfo.observation) + 4));
	ifinfo.basefilename[strlen(argv[1]) - (strlen(ifinfo.observation) + 4)] = 0x0;
	
	if(strstr(ifinfo.observation, ".fits") != NULL) memset(ifinfo.observation + strlen(ifinfo.observation) - 5, 0x0, 5);

	
	if(ifinfo.observation == NULL || ifinfo.basefilename == NULL) {
		printf("No input stem specified!\n");
		exit(1);
	}

	if(argc == 3) {
	strcpy(sf.basefilename, argv[2]);
	} else {
	strcpy(sf.basefilename, ifinfo.basefilename);	
	}
	
	//sscanf(argv[1], "%s/IF%s.fits",sf.basefilename, sf.filename);	
    //sprintf(sf.basefilename, strtok(argv[1], "IF/"));
    //sprintf(sf.filename, strtok(argv[1], ".fits"));

	printf("project directory: %s observation: %s\n", ifinfo.basefilename, ifinfo.observation);



	
	/* set some psrfits params */
	pf.rows_per_file = ROWS_PER_FILE;
	


	int spectracnt = 0;
	int status;


	float *specdata;			
	float *subintdata;
	float *medians;
	float *mads;
	float *means;
				 			
	

	 int nhdu, hdutype, nkeys;
  	 long nrows;
	 int last_fpgactr = 0; 

	  double temp_dbl;
  
	  char *name[2];
	  int ii;

	    for (ii = 0; ii < 2; ii++)    /* allocate space for string column value */
	        name[ii] = (char *) malloc(16);
	

	/* open IF/<observation>.fits and determine number of VEGAS banks used, grab tuning info */
	for (ii = 0; ii < 8; ii++)    /* allocate space for string column value */
	    ifinfo.bank[ii] = (char *) calloc(16, sizeof(char));

	sprintf(ifinfo.filename, "%s/IF/%s.fits",ifinfo.basefilename,ifinfo.observation);    
	if(!exists(ifinfo.filename)) {
		fprintf(stderr, "error - can't find %s\n", ifinfo.filename);
		exit(1);
	}


	read_if_params(&ifinfo);
	
	for(j = 0; j < ifinfo.N; j++) {
		printf("%s %f %f %f \n", ifinfo.bank[j], ifinfo.sff_multiplier[j], ifinfo.sff_offset[j], ifinfo.sff_sideband[j]);
	}
	
	
	/* open LO1A/<observation>.fits and grab LO value */

	sprintf(ifinfo.filename, "%s/LO1A/%s.fits",ifinfo.basefilename,ifinfo.observation);  
	fprintf(stderr, "%s\n", ifinfo.filename);
	if(!exists(ifinfo.filename)) {
		fprintf(stderr, "error - can't find %s\n", ifinfo.filename);
		exit(1);
	}

	
	read_lo_params(&ifinfo);
	
	for(j = 0; j < ifinfo.N; j++) {
		printf("%s %f %f %f %f\n", ifinfo.bank[j], ifinfo.sff_multiplier[j], ifinfo.sff_offset[j], ifinfo.sff_sideband[j], ifinfo.lo);
	}



	/* open GO/<observation>.fits */
	
	sprintf(ifinfo.filename, "%s/GO/%s.fits",ifinfo.basefilename,ifinfo.observation);    
	if(!exists(ifinfo.filename)) {
		fprintf(stderr, "error - can't find %s - hacking in values\n", ifinfo.filename);
		sf.data_columns.ra = 305.654416667;
		sf.data_columns.dec = 28.9064166667;
		sprintf(sf.data_columns.object, "Unknown");
		sprintf(sf.hdr.frontend, "Unknown");
		sprintf(sf.hdr.projid, "Unknown");

	} else {

		read_go_params(&ifinfo, &vf, &sf);
	
	}
	printf("%s %s %f %f\n",sf.data_columns.object,sf.hdr.frontend, sf.data_columns.ra,sf.data_columns.dec);
    
    


    /* Now move on to the data! - we'll eventually want to loop over all the bands in this loop */

float integrat[2];
double utcdelta;
struct stat st;
long int size=0;

    ifinfo.currentbank = 0;
	ifinfo.writtenbank = 0;

for(ifinfo.currentbank = 0; ifinfo.currentbank < ifinfo.N; ifinfo.currentbank = ifinfo.currentbank + 1) {

			sf.N = 0L;
			sf.T = 0.0;
			
			sf.filenum = 1;
			sf.rownum=1;
			
			pf.tot_rows = 0;
			sf.tot_rows = 0;
		
			pf.N = 0L;
			pf.filenum = 0;
			pf.rownum = 1;
			spectracnt = 0;

    
			sprintf(sf.filename, "%s/VEGAS/%s%s.fits",sf.basefilename,ifinfo.observation, ifinfo.bank[ifinfo.currentbank]);    
			while(!exists(sf.filename)) {
				fprintf(stderr, "error - can't find %s\n", sf.filename);
				ifinfo.currentbank = ifinfo.currentbank + 1;
				sprintf(sf.filename, "%s/VEGAS/%s%s.fits",sf.basefilename,ifinfo.observation, ifinfo.bank[ifinfo.currentbank]);    
				if(ifinfo.currentbank == ifinfo.N) exit(1);
			}
		 
			/* not sure if this is initialized to zero elsewhere */
			sf.multifile = 0;
			size=0;
			if(exists(sf.filename)) { 
				stat(sf.filename, &st);
				size = size + st.st_size;
				sf.multifile = 1;
				sf.data_columns.integ_num = 0;
			}
		
			/* we'll use multifile to count sdfits files - not sure if this is consistent with other vegas code */
			printf("File count is %i  Total size is %ld bytes currentbank is %d\n",sf.multifile, size, ifinfo.currentbank);
		
			read_machine_params(&ifinfo, &vf, &sf);
			
			printf("read params..\n");
		
		
		while(sf.filenum <= sf.multifile ) {
		
			 if(sf.rownum == 1) {
		
				  fits_open_file(&sf.fptr, sf.filename, READONLY, &status);
				      if (status) {
						  fprintf(stderr, "Error opening sdfits.\n");
						  fits_report_error(stderr, status);
						  exit(1);
					  }
			
				  /* Move to correct HDU - don't assume anything about EXTVERs */
				  fits_get_num_hdus(sf.fptr, &nhdu, &status);
		
				  fits_movabs_hdu(sf.fptr, 1, &hdutype, &status);
					  
				  /* get header into the hdr buffer */
				  if( fits_hdr2str(sf.fptr, 0, NULL, 0, &hdr, &nkeys, &status ) )
				  printf(" Error getting header\n");
				  
				  fits_movabs_hdu(sf.fptr, 6, &hdutype, &status);
				  
				  fprintf(stderr, "status: %d %d\n",status, hdutype);
				  fits_get_num_rows(sf.fptr, &nrows,  &status);
			
				  fprintf(stderr, "number of rows: %d %ld\n",status, nrows);
				  sf.hdr.nrows = nrows;
			 
				  if(pf.tot_rows == 0 && specdata == NULL) {
						 /* we'll assume we always get AABB from VEGAS machinefits */
						 sf.data_columns.data  = (unsigned char *)malloc(2048 * 4);
						 specdata = malloc(sizeof(float) *  sf.hdr.nchan * 4);
						 memset(specdata, 0x0, sizeof(float) *  sf.hdr.nchan * 4);
					
								
						 /* hack in new nchan */
						 if(CHANEND != 0 && CHANSTART != 0) {
							pf.hdr.nchan = (CHANEND - CHANSTART + 1); 
							pf.hdr.npol = 2;
		
						 } else {
							pf.hdr.nchan = sf.hdr.nchan;
							pf.hdr.npol = 2;
						 }
		
						 pf.sub.bytes_per_subint = pf.hdr.nchan * pf.hdr.npol * SPECTRA_PER_ROW;
		
						 subintdata = (float *) malloc(sizeof(float) * pf.sub.bytes_per_subint); 
						 memset(subintdata, 0x0, sizeof(float) * pf.sub.bytes_per_subint);
		
						 //quantbytes = (unsigned char *) malloc(sf.hdr.nrows * sf.hdr.nchan * pf.hdr.npol); 
								   
						 pf.sub.dat_freqs = (float *)malloc(sizeof(float) * pf.hdr.nchan);
						 pf.sub.dat_weights = (float *)malloc(sizeof(float) * pf.hdr.nchan);
						 pf.sub.dat_offsets = (float *)malloc(sizeof(float) * pf.hdr.nchan * pf.hdr.npol);
						 pf.sub.dat_scales  = (float *)malloc(sizeof(float) * pf.hdr.nchan * pf.hdr.npol);
						 pf.sub.data  = (unsigned char *)malloc(pf.sub.bytes_per_subint);  
						 
						 medians = malloc(sizeof(float) *  pf.hdr.nchan * pf.hdr.npol);
						 mads = malloc(sizeof(float) *  pf.hdr.nchan * pf.hdr.npol);
						 means = malloc(sizeof(float) *  pf.hdr.nchan * pf.hdr.npol);
		
				  } 
		
		
			 }
		
			
		
			/* double dmmjd */
			/* real x 2 integrat */
			/* real * nchan * npol data */
			/* double utcdelta */
			
			
			fits_read_col(sf.fptr, TDOUBLE, 1, sf.rownum, 1, 1, NULL, &(sf.data_columns.time), 
					NULL, &status);
		 
			fits_read_col(sf.fptr, TFLOAT, 2, sf.rownum, 1, 2, NULL, integrat, 
					NULL, &status);
					
			fits_read_col(sf.fptr, TFLOAT, 3, sf.rownum, 1, sf.hdr.nchan * sf.hdr.npol, NULL, specdata,
					NULL, &status);
		
			fits_read_col(sf.fptr, TDOUBLE, 4, sf.rownum, 1, 1, NULL, &utcdelta, 
					NULL, &status);
			
		
			//printf("%f %f\n", sf.data_columns.time, utcdelta);	
			//for (i = 0; i < 1024; i++) printf("%d: %f %f\n", i, specdata[i], specdata[i+1024]);
			
			//exit(0);
		
			
			//fprintf(stderr, "%d integnum: %d accumid: %d %f sttspec: %d stpspec: %d az: %f chan_bw: %f, exposure: %f\n",status, sf.data_columns.integ_num, sf.data_columns.accumid, sf.data_columns.centre_freq[0], sf.data_columns.sttspec, sf.data_columns.stpspec, sf.data_columns.azimuth, sf.hdr.chan_bw, sf.data_columns.exposure);
		
		
			strcpy(sf.data_columns.object, name[0]);
		
		
			if(CHANEND != 0 && CHANSTART != 0) {
				   for(j = CHANSTART; j <= CHANEND; j=j+1) {
					   subintdata[( (spectracnt)*pf.hdr.nchan*pf.hdr.npol) + (j - CHANSTART)] = specdata[(j)];
					   subintdata[( (spectracnt)*pf.hdr.nchan*pf.hdr.npol) + pf.hdr.nchan + (j - CHANSTART)] = specdata[(j + sf.hdr.nchan)];
				   }
			
			} else {
				   for(j = 0;j < sf.hdr.nchan;j=j+1) {
					   //subintdata[((spectracnt)*sf.hdr.nchan*pf.hdr.npol) + (j*pf.hdr.npol)] = specdata[(j * 4)];
					   //subintdata[((spectracnt)*sf.hdr.nchan*pf.hdr.npol) + (j*pf.hdr.npol) + 1] = specdata[(j * 4) + 1];
					   subintdata[( (spectracnt)*pf.hdr.nchan*pf.hdr.npol) + j] = specdata[(j)];
					   subintdata[( (spectracnt)*pf.hdr.nchan*pf.hdr.npol) + pf.hdr.nchan + j] = specdata[(j + sf.hdr.nchan)];
				   }
			}
		
		
		
			spectracnt++;
		
			sf.rownum++;
			sf.tot_rows++;
		
			sf.N++;
			
			//fprintf(stderr, "%d %f integnum: %d accumid: %d %f sttspec: %d stpspec: %d az: %f chan_bw: %f, exposure: %f\n",status, sf.hdr.chan_bw, sf.data_columns.integ_num, sf.data_columns.accumid, sf.data_columns.centre_freq[0], sf.data_columns.sttspec, sf.data_columns.stpspec, sf.data_columns.azimuth, sf.hdr.chan_bw, sf.data_columns.exposure);
		
		
			
			
			if(spectracnt == SPECTRA_PER_ROW) {
				/* write subint */
		
				if(pf.filenum == 0) {
		
					if(argc == 6) {
						sf.hdr.hwexposr = atof(argv[2]);
					} else {
						//if(sf.data_columns.time_counter < 0) sf.data_columns.time_counter = sf.data_columns.time_counter + 4294967296;
						//if(last_fpgactr < 0) last_fpgactr = last_fpgactr + 4294967296;
						//sf.hdr.hwexposr = fabs((double) (sf.data_columns.time_counter - last_fpgactr)) / (double) sf.hdr.fpgaclk;			
						sf.hdr.hwexposr = (double) (1.0/2880000000.0) * 768.0 * 2048.0;
		
						//fprintf(stderr, "calculated expos time: %d %d %f %15.15f\n",sf.data_columns.time_counter, last_fpgactr, sf.hdr.fpgaclk, sf.hdr.hwexposr);			
					}
					//fprintf(stderr, "center freq: %f idindx: %f CP: %d integnum: %d accumid: %d %f sttspec: %d stpspec: %d az: %f chan_bw: %f, exposure: %f\n",sf.data_columns.centre_freq[0], sf.data_columns.centre_freq_idx,status, sf.data_columns.integ_num, sf.data_columns.accumid, sf.data_columns.centre_freq[0], sf.data_columns.sttspec, sf.data_columns.stpspec, sf.data_columns.azimuth, sf.hdr.chan_bw, sf.data_columns.exposure);
					//sf.hdr.obsfreq = (double) sf.data_columns.centre_freq_idx;
		
					/* create the psrfits file and header */
					sdfits_to_psrfits_create(&sf, &pf, &ifinfo);
					pf.filenum = 1;
		
				}
		 
				if(pf.tot_rows == 0) {
					//fprintf(stderr, "computing stats on %d channels and %d pols\n", pf.hdr.nchan, pf.hdr.npol);
					compute_stat(subintdata, pf.hdr.nchan, pf.hdr.npol, SPECTRA_PER_ROW, medians, mads, means);
				} else if (CONTINUOUS_QUANT) {
					//fprintf(stderr, "computing stats on %d channels and %d pols\n", pf.hdr.nchan, pf.hdr.npol);
					compute_stat(subintdata, pf.hdr.nchan, pf.hdr.npol, SPECTRA_PER_ROW, medians, mads, means);		
				} else {
					massage(subintdata, pf.hdr.nchan, pf.hdr.npol, SPECTRA_PER_ROW, medians, mads, means);					
				}
		
					 
				//fprintf(stderr, "000000 %f %f %f\n", mads[500], medians[500], means[500]);
				//fprintf(stderr, "quantizing\n");
				quant(subintdata, pf.sub.data, pf.hdr.nchan, pf.hdr.npol, SPECTRA_PER_ROW, 6, mads);
		
				//for(j = 0; j < sf.hdr.nchan * pf.hdr.npol; j++) pf.sub.dat_scales[j] = 1.0;
				for(j = 0; j < pf.hdr.nchan * pf.hdr.npol; j++) pf.sub.dat_scales[j] = (1.4826 * mads[j] * 12)/256;
		
				for(j = 0; j < pf.hdr.nchan * pf.hdr.npol; j++) pf.sub.dat_weights[j] = 1.0;
		
				/* set weights for the fake pulsar */
				//for(j = 0; j < pf.hdr.nchan * pf.hdr.npol; j++) pf.sub.dat_weights[j] = 0.0;
				//for(j = 5; j < 125; j++) pf.sub.dat_weights[j] = 1.0;
				
					
				for(j = 0; j < pf.hdr.nchan * pf.hdr.npol; j++) pf.sub.dat_offsets[j] = medians[j];
				for(j = 0; j < pf.hdr.nchan; j++) pf.sub.dat_freqs[j] = (pf.hdr.fctr - (pf.hdr.df * (pf.hdr.nchan / 2)) + ((double) j * pf.hdr.df));		
				
				//fprintf(stderr, "writing subint\n");
				sdfits_to_psrfits_write_subint(&sf, &pf);
				spectracnt = 0;
			}
		
			if(sf.tot_rows == sf.hdr.nrows) {
				/* end of file, open a new one */
				sf.tot_rows = 0;
				sf.rownum = 1;
				sf.filenum++;	
				fits_close_file(sf.fptr, &status);
			}
		
			last_fpgactr = sf.data_columns.time_counter;
			sf.data_columns.integ_num++;
			
		
		}
		
		
			fprintf(stderr, "%d %f integnum: %d\n", status, sf.hdr.chan_bw, sf.data_columns.integ_num);
			fprintf(stderr, "\n");
			
			//free(subintdata);
		
			/* close sdfits input */
			//fits_close_file(sf.fptr, &status);
		
		    if (status) {
				fprintf(stderr, "Error closing file.\n");
				fits_report_error(stderr, status);
				exit(1);
			}
			
			/* flush and close psrfits output */
			fits_flush_file(pf.fptr, &status);
			    if (status) {
					fprintf(stderr, "Error flushing.\n");
					fits_report_error(stderr, status);
					exit(1);
				}


			fits_close_file(pf.fptr, &status);
			    if (status) {
					fprintf(stderr, "Error closing.\n");
					fits_report_error(stderr, status);
					exit(1);
				}
		
		

	}	
			
	
	exit(0);		 	
	
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



// Read a status buffer all of the key observation paramters
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
        fprintf(stderr, "Error creating psrfits file from template file.\n");
        fits_report_error(stderr, status);
        exit(1);
    }	

		
	//f_sky = SFF_SIDEBAND * IF + +SFF_MULTIPLIER*LO1 + SFF_OFFSET
	//For a given SUBBAND, IF = CRVAL1 + i * CDELT1 where i goes from 0 to
	//(number of channels-1).

	
}






void bin_print_verbose(short x)
/* function to print decimal numbers in verbose binary format */
/* x is integer to print, n_bits is number of bits to print */
{

   int j;
   printf("no. 0x%08x in binary \n",(int) x);

   for(j=16-1; j>=0;j--){
	   printf("bit: %i = %i\n",j, (x>>j) & 01);
   }

}


void massage(float *data, int nchan, int npol, int nframe, float *medians, float *mads, float *means) {

	 long int ii,jj;
	 double sum = 0 ;
	 for(ii = 0;ii<(nchan * npol);ii = ii + 1){		  
	 	  for (jj = 0; jj < nframe; jj = jj + 1) {
				data[(jj * nchan * npol) + ii] = data[(jj * nchan * npol) + ii] + 0.00001 - medians[ii];
		  		//sum = sum + data[(jj * nchan * npol) + ii]; 
	 	  }			
	 }

}



void compute_stat(float *data, int nchan, int npol, int nframe, float *medians, float *mads, float *means) {

	 long int ii,jj;
	 float *tempvec;
	 tempvec = (float *) malloc(sizeof(float) * nframe);
	 double sum = 0;	 	 
	 

	 for(ii = 0;ii<(nchan * npol);ii = ii + 1){		  
	 	  sum = 0;
	 	  for (jj = 0; jj < nframe; jj = jj + 1) {
				data[(jj * nchan * npol) + ii] = data[(jj * nchan * npol) + ii] + 0.00001;
 		  		tempvec[jj] =  data[(jj * nchan * npol) + ii];
				sum = sum + (double) tempvec[jj];				

	 	  }
		
		  sum = sum / nframe;
		  means[ii] = (float) sum;
		  medians[ii] = median(tempvec, nframe);
		  

	 	  for (jj = 0; jj < nframe; jj = jj + 1) {
				tempvec[jj] =  fabsf(data[(jj * nchan * npol) + ii] - medians[ii]);	 	  		
 		  		data[(jj * nchan * npol) + ii] = data[(jj * nchan * npol) + ii] - medians[ii];
	 	  }
	 	  
		  mads[ii] = median(tempvec, nframe);
		  if(fabsf(mads[ii]) < 0.0001) mads[ii] = 1.0;
	 }

	 free(tempvec);		  		
}



void quant(float *data, unsigned char * quantbytes, int nchan, int npol, int nframe, int sigma, float *mads) {

	long int ii,jj;
	float min;
	float max;
	
		
	 for (jj = 0; jj < nframe; jj = jj + 1) {
		  for(ii = 0;ii<(nchan * npol);ii = ii + 1){		  
			  quantbytes[(jj * nchan * npol) + ii] = uquantize(data[(jj * nchan * npol) + ii], -1.4826 * mads[ii] * sigma, 1.4826 * mads[ii] * sigma);
			  //min =  -1.4826 * mads[ii] * sigma;
			  //max = 1.4826 * mads[ii] * sigma;
			  //printf("%f %f %f %f %d\n", data[(jj * nchan * npol) + ii], min, max, (int)(((data[(jj * nchan * npol) + ii] - min) / (max-min)) * 255.0));
		  }
	 }
 //usleep(500000000);

}



unsigned char uquantize(float d, float min, float max)
{
    if(d > max) d = max;
    if(d < min) d = min;
    
 	return (unsigned char)( ((d - min) / (max-min)) * 255.0);
}





int sdfits_to_psrfits_create(struct sdfits *sf, struct psrfits *pf, struct iffits *ifinfo) {
    int itmp, *status;
    long double ldtmp;
    double dtmp;
    char ctmp[40];
    struct sdfits_hdrinfo *hdr;
	struct sdfits_data_columns *dcols;
	
	sf->status = 0;
    hdr = &(sf->hdr);        // dereference the ptr to the header struct
    status = &(sf->status);  // dereference the ptr to the CFITSIO status
    dcols = &(sf->data_columns);    // dereference the ptr to the subint struct
	
	char tempfilname[200];
	if(!exists(ifinfo->observation)) mkdir(ifinfo->observation, 0777); 
	sprintf(tempfilname, "%s/vegas-hpc%d-bdata1", ifinfo->observation, ifinfo->writtenbank);
	mkdir(tempfilname, 0777);
	sprintf(tempfilname, "%s/vegas-hpc%d-bdata1/psr_%s_0001.fits", ifinfo->observation, ifinfo->writtenbank, ifinfo->observation);
	strcpy(pf->filename, tempfilname);

	printf("filename: %s\n",pf->filename);

    char *guppi_dir = getenv("GUPPI_DIR");
    char template_file[1024];
    if (guppi_dir==NULL) {
        fprintf(stderr, 
                "Error: GUPPI_DIR environment variable not set, exiting.\n");
        exit(1);
    }
    printf("Opening file '%s' for writing up to %d rows\n", pf->filename, pf->rows_per_file);

    sprintf(template_file, "%s/%s", guppi_dir, PSRFITS_SEARCH_TEMPLATE);
    printf("using: %s\n", template_file);
    fits_create_template(&(pf->fptr), pf->filename, template_file, status);

    printf("created psrfits\n");
	ifinfo->writtenbank++;
	fflush(stdout);
    // Check to see if file was successfully created
    if (*status) {
        fprintf(stderr, "Error creating psrfits file from template file.\n");
        fits_report_error(stderr, *status);
        exit(1);
    }
 


	pf->hdr.df = hdr->chan_bw/1000000;
	
	
	//if(hdr->obsfreq < 100000) hdr->obsfreq = hdr->obsfreq * 1000000;

	if(CHANEND != 0 && CHANSTART != 0) {
		//+ (((double) (CHANEND - CHANSTART + 1) * hdr->chan_bw) ) /(1000000 * 2))
		/* new center frequency */
		pf->hdr.fctr = (hdr->obsfreq - (hdr->chan_bw * (hdr->nchan / 2)) + ((double) CHANSTART * hdr->chan_bw) ) /1000000  - (hdr->chan_bw/(2 * 1000000)) ;

		/* new observation bandwidth */
	    pf->hdr.BW = fabs(hdr->chan_bw/1000000 * (double) (CHANEND - CHANSTART + 1));

	    pf->hdr.fctr = pf->hdr.fctr + pf->hdr.BW/2; 

	} else {
	
		/* take these values from the sdfits header */
    	pf->hdr.BW = hdr->chan_bw * hdr->nchan;
		pf->hdr.fctr = hdr->obsfreq/1000000;
	
	}



	
	printf("sdfits center freq: %f sdfits chanbw %f psrfits center freq %f bandwidth %f...\n", hdr->obsfreq, hdr->chan_bw, pf->hdr.fctr, pf->hdr.BW);

	printf("hacking npol to 2...\n");
	hdr->npol = 2;
	printf("exposure: %f\n",hdr->hwexposr);

	/* FIXED: now set hwexposr before psrfits file creation based on fpga ctr and fpga clock freq */
	/* hwexposr isn't properly set in sdfits headers for mode 1 - we'll recode it to a datacol exposure len */
	//hdr->hwexposr = dcols->exposure;
	
	
    // Go to the primary HDU
    fits_movabs_hdu(pf->fptr, 1, NULL, status);

    // Update the keywords that need it
    fits_get_system_time(ctmp, &itmp, status);
    // Note:  this is the date the file was _written_, not the obs start date
    fits_update_key(pf->fptr, TSTRING, "DATE", ctmp, NULL, status);
    fits_update_key(pf->fptr, TSTRING, "TELESCOP", hdr->telescope,NULL, status);
    strcpy(ctmp, "WAGNER");
    fits_update_key(pf->fptr, TSTRING, "OBSERVER", ctmp, NULL, status);
    fits_update_key(pf->fptr, TSTRING, "PROJID", hdr->projid, NULL, status);
    fits_update_key(pf->fptr, TSTRING, "FRONTEND", hdr->frontend, NULL, status);
    strcpy(ctmp, "VEGAS");
    fits_update_key(pf->fptr, TSTRING, "BACKEND", ctmp, NULL, status);

     if (hdr->npol > 2) { // Can't have more than 2 real polns (i.e. NRCVR)
         itmp = 2;
         fits_update_key(pf->fptr, TINT, "NRCVR", &itmp, NULL, status);
     } else {
         fits_update_key(pf->fptr, TINT, "NRCVR", &(hdr->npol), NULL, status);
     }

    //fits_update_key(pf->fptr, TSTRING, "FD_POLN", hdr->poln_type, NULL, status);
    //fits_update_key(pf->fptr, TINT, "FD_HAND", &(hdr->fd_hand), NULL, status);
    fits_update_key(pf->fptr, TSTRING, "DATE-OBS", hdr->date_obs, NULL, status);
    //if (mode==fold && !strcmp("CAL",hdr->obs_mode)) 
    //    fits_update_key(pf->fptr, TSTRING, "OBS_MODE", hdr->obs_mode, 
    //            NULL, status);

	dtmp = pf->hdr.fctr;
    fits_update_key(pf->fptr, TDOUBLE, "OBSFREQ", &dtmp, NULL, status);

    dtmp = pf->hdr.BW;
    fits_update_key(pf->fptr, TDOUBLE, "OBSBW", &dtmp, NULL, status);
    
    itmp = pf->hdr.nchan;
    fits_update_key(pf->fptr, TINT,    "OBSNCHAN", &(itmp), NULL, status);

    dtmp = 0.0;
    fits_update_key(pf->fptr, TDOUBLE, "CHAN_DM", &dtmp, NULL, status);
    strcpy(ctmp, "UNKNOWN");
    fits_update_key(pf->fptr, TSTRING, "SRC_NAME", ctmp, NULL, status);
    //if (!strcmp("UNKNOWN", hdr->track_mode)) {
    //    printf("Warning!:  Unknown telescope tracking mode!\n");
    //}
    //fits_update_key(pf->fptr, TSTRING, "TRK_MODE", hdr->track_mode, NULL, status);
    // TODO: will need to change the following if we aren't tracking!
    
 
 
    
	dec2hms(ctmp, dcols->ra/15.0, 0);
    
    fits_update_key(pf->fptr, TSTRING, "RA", ctmp, NULL, status);
    fits_update_key(pf->fptr, TSTRING, "STT_CRD1", ctmp, NULL, status);
    fits_update_key(pf->fptr, TSTRING, "STP_CRD1", ctmp, NULL, status);

	dec2hms(ctmp, dcols->dec, 1);

    fits_update_key(pf->fptr, TSTRING, "DEC", ctmp, NULL, status);
    fits_update_key(pf->fptr, TSTRING, "STT_CRD2", ctmp, NULL, status);
    fits_update_key(pf->fptr, TSTRING, "STP_CRD2", ctmp, NULL, status);
    
	dcols->bmaj=0.0;
	dcols->bmin=0.0;
	dtmp = (double) dcols->bmaj;
    fits_update_key(pf->fptr, TDOUBLE, "BMAJ", &(dtmp), NULL, status);
	dtmp = (double) dcols->bmin;
    fits_update_key(pf->fptr, TDOUBLE, "BMIN", &(dtmp), NULL, status);



    dtmp = 3600;
    fits_update_key(pf->fptr, TDOUBLE, "SCANLEN", &dtmp, NULL, status);
    //printf("MJD %15.15f\n", hdr->sttmjd);
    if(hdr->sttmjd < 1) hdr->sttmjd  = 56000.5;
    itmp = (int) hdr->sttmjd;
    fits_update_key(pf->fptr, TINT, "STT_IMJD", &itmp, NULL, status);
    /* assume whole second - double not enough for hdr->sttmjd? */
    ldtmp = round(((long double) hdr->sttmjd - (long double) itmp) * 86400.0L);   // in sec
    itmp = (int) ldtmp;
    fits_update_key(pf->fptr, TINT, "STT_SMJD", &itmp, NULL, status);
    ldtmp -= (long double) itmp;
    dtmp = (double) ldtmp;
    fits_update_key(pf->fptr, TDOUBLE, "STT_OFFS", &dtmp, NULL, status);
    dtmp = 0;
    fits_update_key(pf->fptr, TDOUBLE, "STT_LST", &dtmp, NULL, status);


        
    // Go to the SUBINT HDU
    fits_movnam_hdu(pf->fptr, BINARY_TBL, "SUBINT", 0, status);

	
    // Update the keywords that need it
    fits_update_key(pf->fptr, TINT, "NPOL", &(hdr->npol), NULL, status);
 
  
    if (hdr->npol==1)
        strcpy(ctmp, "AA+BB");
    else if (hdr->npol==2)
        strcpy(ctmp, "AABB"); //vegas says 2 pols when full stokes.
    else if (hdr->npol==4)
        strcpy(ctmp, "AABBCRCI");
    fits_update_key(pf->fptr, TSTRING, "POL_TYPE", ctmp, NULL, status);

    // TODO what does TBIN mean in fold mode?
    //dtmp = hdr->dt * hdr->ds_time_fact;
    
	dtmp = 0;
    dtmp = hdr->hwexposr;
    pf->hdr.dt = dtmp;
    printf("exposure: %f\n",dtmp);

    fits_update_key(pf->fptr, TDOUBLE, "TBIN",  &(dtmp), NULL, status);
    
    //fits_update_key(pf->fptr, TINT, "NSUBOFFS", &(hdr->offset_subint), NULL, status);
    itmp = pf->hdr.nchan;
    fits_update_key(pf->fptr, TINT, "NCHAN", &itmp, NULL, status);
    
    //dtmp = hdr->df * hdr->ds_freq_fact;
    //dtmp = hdr->chan_bw/1000000;
    dtmp = pf->hdr.df;
    
    fits_update_key(pf->fptr, TDOUBLE, "CHAN_BW", &dtmp, NULL, status);
   


    int out_nsblk = SPECTRA_PER_ROW;
    fits_update_key(pf->fptr, TINT, "NSBLK", &out_nsblk, NULL, status);
    itmp = 8;
    fits_update_key(pf->fptr, TINT, "NBITS", &(itmp), NULL, status);
    itmp = 1;
    fits_update_key(pf->fptr, TINT, "NBIN", &itmp, NULL, status);

	//itmp = (int) ((float) (dcols->integ_num + 1 - hdr->nrows)  / (float) hdr->nrows);
	itmp = (int) ((float) (dcols->integ_num)  / (float) SPECTRA_PER_ROW);
	
	fits_update_key(pf->fptr, TINT, "NSUBOFFS", &itmp, NULL, status);

	
	
	pf->sub.tsubint = (double) hdr->hwexposr * (double) SPECTRA_PER_ROW;
	
	
	/* the formula below is wrong!  we're not using it so I won't fix it just yet */
	//pf->sub.offs = (((double) (dcols->integ_num + 1 - hdr->nrows)) * hdr->hwexposr) + (0.5 * hdr->hwexposr);
	
    
    // Update the column sizes for the colums containing arrays
    
	int out_npol = pf->hdr.npol;
	int out_nchan = pf->hdr.nchan;
	int out_nbits = 8;

	fits_modify_vector_len(pf->fptr, 13, out_nchan, status); // DAT_FREQ
	fits_modify_vector_len(pf->fptr, 14, out_nchan, status); // DAT_WTS
	itmp = out_nchan * out_npol;
	fits_modify_vector_len(pf->fptr, 15, itmp, status); // DAT_OFFS
	fits_modify_vector_len(pf->fptr, 16, itmp, status); // DAT_SCL
	
	itmp = (out_nbits * out_nchan * out_npol * out_nsblk) / 8;  
	
	fits_modify_vector_len(pf->fptr, 17, itmp, status); // DATA
	// Update the TDIM field for the data column

	sprintf(ctmp, "(1,%d,%d,%d)", out_nchan, out_npol, out_nsblk);
	fits_update_key(pf->fptr, TSTRING, "TDIM17", ctmp, NULL, status);

    fits_flush_file(pf->fptr, status);

    if (*status) {
        fprintf(stderr, "Error creating psrfits\n");
        fits_report_error(stderr, *status);
        fflush(stderr);
    }
    
    return *status;
}


int sdfits_to_psrfits_write_subint(struct sdfits *sf, struct psrfits *pf) {
    int *status;
    float ftmp;
    struct sdfits_hdrinfo *hdr;
    struct sdfits_data_columns *dcols;
    double dtmp;



    hdr = &(sf->hdr);        // dereference the ptr to the header struct
    dcols = &(sf->data_columns);        // dereference the ptr to the subint struct
    status = &(sf->status);  // dereference the ptr to the CFITSIO status


    //mode = psrfits_obs_mode(hdr->obs_mode);

	    
	//dtmp = (double) dcols->exposure;
	//fprintf(stderr, "%g\n", dtmp);

	dtmp = pf->sub.tsubint;
    fits_write_col(pf->fptr, TDOUBLE, 1, pf->rownum, 1, 1, &(dtmp), status); //

	//dtmp = pf->sub.offs;
	//dtmp = (((double) (dcols->integ_num + 1 - hdr->nrows)) * hdr->hwexposr) + (0.5 * hdr->hwexposr);
	dtmp = ((double) SPECTRA_PER_ROW * (double) (pf->rownum - 1) * pf->hdr.dt) +  ((double) SPECTRA_PER_ROW * 0.5 * pf->hdr.dt);

    fits_write_col(pf->fptr, TDOUBLE, 2, pf->rownum, 1, 1, &(dtmp), status); //

	//lst
	dtmp = 0;
    fits_write_col(pf->fptr, TDOUBLE, 3, pf->rownum, 1, 1, &dtmp, status);

    fits_write_col(pf->fptr, TDOUBLE, 4, pf->rownum, 1, 1, &(dcols->ra), status); //
    fits_write_col(pf->fptr, TDOUBLE, 5, pf->rownum, 1, 1, &(dcols->dec), status); //

	/* galactic lat/long */
	dtmp = 0;
    fits_write_col(pf->fptr, TDOUBLE, 6, pf->rownum, 1, 1, &dtmp, status);
    fits_write_col(pf->fptr, TDOUBLE, 7, pf->rownum, 1, 1, &dtmp, status);

    ftmp = 0.0;
    fits_write_col(pf->fptr, TFLOAT, 8, pf->rownum, 1, 1, &ftmp, status);
    fits_write_col(pf->fptr, TFLOAT, 9, pf->rownum, 1, 1, &ftmp, status);
    fits_write_col(pf->fptr, TFLOAT, 10, pf->rownum, 1, 1, &ftmp, status);
    
    dcols->azimuth = 0;
    dcols->elevation = 0;
    fits_write_col(pf->fptr, TFLOAT, 11, pf->rownum, 1, 1, &(dcols->azimuth), status); //
    fits_write_col(pf->fptr, TFLOAT, 12, pf->rownum, 1, 1, &(dcols->elevation), status); //
    
    
    
    fits_write_col(pf->fptr, TFLOAT, 13, pf->rownum, 1, pf->hdr.nchan, pf->sub.dat_freqs, status);
    fits_write_col(pf->fptr, TFLOAT, 14, pf->rownum, 1, pf->hdr.nchan, pf->sub.dat_weights, status);
    fits_write_col(pf->fptr, TFLOAT, 15, pf->rownum, 1, pf->hdr.nchan*2, pf->sub.dat_offsets, status);
    fits_write_col(pf->fptr, TFLOAT, 16, pf->rownum, 1, pf->hdr.nchan*2, pf->sub.dat_scales, status);

	fits_write_col(pf->fptr, TBYTE,  17, pf->rownum, 1, pf->sub.bytes_per_subint, pf->sub.data, status);


    // Print status if bad
    if (*status) {
        fprintf(stderr, "Error writing subint %d:\n", pf->rownum);
        fits_report_error(stderr, *status);
        fflush(stderr);
        exit(1);
    }
    
//    fits_write_col(pf->fptr, TBYTE, 17, row, 1, out_nbytes, 
//                       sub->data, status);

    // Flush the buffers if not finished with the file
    // Note:  this use is not entirely in keeping with the CFITSIO
    //        documentation recommendations.  However, manually 
    //        correcting NAXIS2 and using fits_flush_buffer()
    //        caused occasional hangs (and extrememly large
    //        files due to some infinite loop).
    fits_flush_file(pf->fptr, status);


    // Now update some key values if no CFITSIO errors
    if (!(*status)) {


        pf->rownum++;
        pf->tot_rows++;

        pf->N += SPECTRA_PER_ROW;
    }

    return *status;
}

int exists(const char *fname)
{
    FILE *file;
    if ((file = fopen(fname, "r")))
    {
        fclose(file);
        return 1;
    }
    return 0;
}
