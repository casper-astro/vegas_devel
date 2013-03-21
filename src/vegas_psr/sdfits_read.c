#include <stdio.h>
#include <stdlib.h>
#include "fitsio.h"
#include "psrfits.h"
#include "vegas_params.h"
#include "fitshead.h"
#include <math.h>
#include <arpa/inet.h>
#include <string.h>
#include "median.h"
#include <sys/stat.h>

/* set to 1 to recalculate stats continuously */
#define CONTINUOUS_QUANT 1

/* number of spectra to put in each subint */
#define SPECTRA_PER_ROW 16384

/* total number of subints per file */
#define ROWS_PER_FILE 1000000

/* first channel to take */
#define CHANSTART 72

/*FAKE PULSAR TEST*/
//#define CHANSTART 4

/* last channel to keep */
#define CHANEND 994


/* Parse info from buffer into param struct */
extern void vegas_read_obs_params(char *buf, 
                                     struct vegas_params *g,
                                     struct sdfits *p);
double round(double x);
unsigned char uquantize(float d, float min, float max);             
int exists(const char *fname);
 

int sdfits_to_psrfits_write_subint(struct sdfits *sf, struct psrfits *pf);
int sdfits_to_psrfits_create(struct sdfits *sf, struct psrfits *pf);
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

    char buf[65536];
    char filname[250];
	int filepos=0;
    
    FILE *fil;   //input file
    

	long int i,j,k;
    

    if(argc < 2) {
		fprintf(stderr, "USAGE: %s input.fits (use 'stdout' for output if stdout is desired)\n", argv[0]);
		exit(1);
	}
    

    sprintf(sf.basefilename, argv[1]);


	/* no input specified */
	if(sf.basefilename == NULL) {
		printf("No input stem specified!\n");
		exit(1);
	}
	
	
	
	if(strstr(sf.basefilename, "_0001.fits") != NULL) memset(sf.basefilename + strlen(sf.basefilename) - 10, 0x0, 10);
		
	
	/* set file counter to one file */
	j = 1;
	struct stat st;
	long int size=0;
	do {
		sprintf(filname, "%s_%04ld.fits",sf.basefilename,j);
		printf("%s\n",filname);		
		j++;
		if(exists(filname)) { 
			stat(filname, &st);
			size = size + st.st_size;
		}
	} while (exists(filname));

	/* we'll use multifile to count sdfits files - not sure if this is consistent with other vegas code */
	sf.multifile = j-2;
	printf("File count is %i  Total size is %ld bytes\n",sf.multifile, size);
	
		
	/* didn't find any files */
	if(sf.multifile < 1) {
		printf("no files for stem %s found\n",sf.basefilename);
		exit(1);		
	}






    sf.N = 0L;
    sf.T = 0.0;
	
	sf.filenum = 1;
	sf.rownum=1;
	
    pf.tot_rows = 0;
    sf.tot_rows = 0;

	/* set some psrfits params */
	pf.rows_per_file = ROWS_PER_FILE;
	
	pf.N = 0L;
	pf.filenum = 0;
	pf.rownum = 1;

	int spectracnt = 0;
	int status;


	float *specdata;			
	float *subintdata;
	float *medians;
	float *mads;
	float *means;
				 			 

	

	 int nhdu, hdutype;
  	 long nrows;
	 int last_fpgactr = 0;

	  double temp_dbl;
  
	  char *name[2];
	  int ii;

	    for (ii = 0; ii < 2; ii++)    /* allocate space for string column value */
	        name[ii] = (char *) malloc(16);

while(sf.filenum <= sf.multifile ) {

	 if(sf.rownum == 1) {
		 sprintf(sf.filename, "%s_%04d.fits",sf.basefilename,sf.filenum);    
		 fil = fopen(sf.filename, "rb");
		 
		 
		 filepos=0;
		 
		 fread(buf, sizeof(char), 65536, fil); 		
	 
		 fseek(fil, -65536, SEEK_CUR);
		 //printf("lhead: %d", lhead0);
		 fprintf(stderr, "length: %d\n", gethlength(buf));
		 fprintf(stderr, "length: %d\n", gethlength(buf + (gethlength(buf) + gethlength(buf)%2880)));
		 long int totheadlength;
		 /* get total length of header - main header + bin table */
	 
		 totheadlength = gethlength(buf) + gethlength(buf)%2880 + gethlength(buf + (gethlength(buf) + gethlength(buf)%2880));
		 fprintf(stderr, "length: %ld\n", totheadlength);
	 
	 
		 vegas_read_obs_params(buf, &vf, &sf);
	 
		 
		 fprintf(stderr, "nrows: %d\n", sf.hdr.nrows);    
		 fprintf(stderr, "nwidth: %d\n", sf.hdr.nwidth);    
		 
		 fprintf(stderr, "nchan in sdfits: %d\n", sf.hdr.nchan);    
		 //fprintf(stderr, "size %d\n",sf.sub.bytes_per_subint + gethlength(buf));
		 fprintf(stderr, "bandwidth %f\n", sf.hdr.bandwidth);
		 fprintf(stderr, "freqres %f\n", sf.hdr.freqres);
		 
		 fprintf(stderr, "efsampfr %f\n", sf.hdr.efsampfr);
		 fprintf(stderr, "fpgaclk %f\n", sf.hdr.fpgaclk);
		 
		 fclose(fil);
	 
		 fits_open_file(&sf.fptr, sf.filename, READONLY, &status);
	 
		  if (status) { /* error flag set - do nothing */
			 fprintf(stderr, "%s %d error\n", argv[1], status);
			 exit(0);
		  }
		  
		  /* Move to correct HDU - don't assume anything about EXTVERs */
		  fits_get_num_hdus(sf.fptr, &nhdu, &status);
		  fprintf(stderr, "nhdus: %d\n",nhdu);
	 
		 
		  fits_movabs_hdu(sf.fptr, 2, &hdutype, &status);
		  fprintf(stderr, "status: %d %d\n",status, hdutype);
		  fits_get_num_rows(sf.fptr, &nrows,  &status);
		  fprintf(stderr, "number of rows: %d %ld\n",status, nrows);
	 
		  if(pf.tot_rows == 0) {
				 /* we'll assume we always get AABBCRCI from VEGAS simonfits */
				 sf.data_columns.data  = (unsigned char *)malloc(4096 * 4);
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

	


   

//	for(i = 1; i < (1 +  sf.hdr.nrows); i++) {


	
	//fprintf(stderr, "object: %s\n",sf.data_columns.object);
	
    fits_read_col(sf.fptr, TDOUBLE, 1, sf.rownum, 1, 1, NULL, &(sf.data_columns.time), 
            NULL, &status);
 
    fits_read_col(sf.fptr, TINT, 2, sf.rownum, 1, 1, NULL, &(sf.data_columns.time_counter), 
            NULL, &status);

    fits_read_col(sf.fptr, TINT, 3, sf.rownum, 1, 1, NULL, &(sf.data_columns.integ_num), 
            NULL, &status);

    fits_read_col(sf.fptr, TFLOAT, 4, sf.rownum, 1, 1, NULL, &(sf.data_columns.exposure), 
            NULL, &status);

    fits_read_col(sf.fptr, TSTRING, 5, sf.rownum, 1, 1, NULL, name, 
            NULL, &status);    
    
    fits_read_col(sf.fptr, TFLOAT, 6, sf.rownum, 1, 1, NULL, &(sf.data_columns.azimuth), 
            NULL, &status);
    fits_read_col(sf.fptr, TFLOAT, 7, sf.rownum, 1, 1, NULL, &(sf.data_columns.elevation), 
            NULL, &status);
    fits_read_col(sf.fptr, TFLOAT, 8, sf.rownum, 1, 1, NULL, &(sf.data_columns.bmaj), 
            NULL, &status);
    fits_read_col(sf.fptr, TFLOAT, 9, sf.rownum, 1, 1, NULL, &(sf.data_columns.bmin), 
            NULL, &status);
    fits_read_col(sf.fptr, TFLOAT, 10, sf.rownum, 1, 1, NULL, &(sf.data_columns.bpa), 
            NULL, &status);
    fits_read_col(sf.fptr, TINT, 11, sf.rownum, 1, 1, NULL, &(sf.data_columns.accumid), 
            NULL, &status);
    fits_read_col(sf.fptr, TINT, 12, sf.rownum, 1, 1, NULL, &(sf.data_columns.sttspec), 
            NULL, &status);
    fits_read_col(sf.fptr, TINT, 13, sf.rownum, 1, 1, NULL, &(sf.data_columns.stpspec),
            NULL, &status);
    fits_read_col(sf.fptr, TFLOAT, 14, sf.rownum, 1, 1, NULL, &(sf.data_columns.centre_freq_idx), 
            NULL, &status);            
    fits_read_col(sf.fptr, TFLOAT, 15, sf.rownum, 1, 1, NULL, sf.data_columns.centre_freq, 
            NULL, &status);
    fits_read_col(sf.fptr, TDOUBLE, 16, sf.rownum, 1, 1, NULL, &temp_dbl, 
    		NULL, &status);            
    fits_read_col(sf.fptr, TDOUBLE, 17, sf.rownum, 1, 1, NULL, &(sf.hdr.chan_bw), 
    		NULL, &status);
    fits_read_col(sf.fptr, TDOUBLE, 18, sf.rownum, 1, 1, NULL, &(sf.data_columns.ra), 
            NULL, &status);
    fits_read_col(sf.fptr, TDOUBLE, 19, sf.rownum, 1, 1, NULL, &(sf.data_columns.dec),
            NULL, &status);         
    fits_read_col(sf.fptr, TFLOAT, 20, sf.rownum, 1, 4096, NULL, specdata,
            NULL, &status);
	
	//fprintf(stderr, "%d integnum: %d accumid: %d %f sttspec: %d stpspec: %d az: %f chan_bw: %f, exposure: %f\n",status, sf.data_columns.integ_num, sf.data_columns.accumid, sf.data_columns.centre_freq[0], sf.data_columns.sttspec, sf.data_columns.stpspec, sf.data_columns.azimuth, sf.hdr.chan_bw, sf.data_columns.exposure);


	strcpy(sf.data_columns.object, name[0]);


	if(CHANEND != 0 && CHANSTART != 0) {
		   for(j = CHANSTART; j <= CHANEND; j=j+1) {
			   subintdata[( (spectracnt)*pf.hdr.nchan*pf.hdr.npol) + (j - CHANSTART)] = specdata[(j * 4)];
			   subintdata[( (spectracnt)*pf.hdr.nchan*pf.hdr.npol) + pf.hdr.nchan + (j - CHANSTART)] = specdata[(j * 4) + 1];
		   }
	
	} else {
		   for(j = 0;j < sf.hdr.nchan;j=j+1) {
			   //subintdata[((spectracnt)*sf.hdr.nchan*pf.hdr.npol) + (j*pf.hdr.npol)] = specdata[(j * 4)];
			   //subintdata[((spectracnt)*sf.hdr.nchan*pf.hdr.npol) + (j*pf.hdr.npol) + 1] = specdata[(j * 4) + 1];
			   subintdata[( (spectracnt)*pf.hdr.nchan*pf.hdr.npol) + j] = specdata[(j * 4)];
			   subintdata[( (spectracnt)*pf.hdr.nchan*pf.hdr.npol) + pf.hdr.nchan + j] = specdata[(j * 4) + 1];
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

			if(argc == 3) {
				sf.hdr.hwexposr = atof(argv[2]);
			} else {
				if(sf.data_columns.time_counter < 0) sf.data_columns.time_counter = sf.data_columns.time_counter + 4294967296;
				if(last_fpgactr < 0) last_fpgactr = last_fpgactr + 4294967296;
				sf.hdr.hwexposr = fabs((double) (sf.data_columns.time_counter - last_fpgactr)) / (double) sf.hdr.fpgaclk;			
			    fprintf(stderr, "calculated expos time: %d %d %f %15.15f\n",sf.data_columns.time_counter, last_fpgactr, sf.hdr.fpgaclk, sf.hdr.hwexposr);			
			}
			fprintf(stderr, "center freq: %f idindx: %f CP: %d integnum: %d accumid: %d %f sttspec: %d stpspec: %d az: %f chan_bw: %f, exposure: %f\n",sf.data_columns.centre_freq[0], sf.data_columns.centre_freq_idx,status, sf.data_columns.integ_num, sf.data_columns.accumid, sf.data_columns.centre_freq[0], sf.data_columns.sttspec, sf.data_columns.stpspec, sf.data_columns.azimuth, sf.hdr.chan_bw, sf.data_columns.exposure);
			sf.hdr.obsfreq = (double) sf.data_columns.centre_freq_idx;

			/* create the psrfits file and header */
			sdfits_to_psrfits_create(&sf, &pf);
			pf.filenum = 1;

		}
	         
		if(pf.tot_rows == 0) {
			fprintf(stderr, "computing stats on %d channels and %d pols\n", pf.hdr.nchan, pf.hdr.npol);
			compute_stat(subintdata, pf.hdr.nchan, pf.hdr.npol, SPECTRA_PER_ROW, medians, mads, means);
		} else if (CONTINUOUS_QUANT) {
			fprintf(stderr, "computing stats on %d channels and %d pols\n", pf.hdr.nchan, pf.hdr.npol);
			compute_stat(subintdata, pf.hdr.nchan, pf.hdr.npol, SPECTRA_PER_ROW, medians, mads, means);		
		} else {
			massage(subintdata, pf.hdr.nchan, pf.hdr.npol, SPECTRA_PER_ROW, medians, mads, means);					
		}

	         
		fprintf(stderr, "000000 %f %f %f\n", mads[500], medians[500], means[500]);
		fprintf(stderr, "quantizing\n");
		quant(subintdata, pf.sub.data, pf.hdr.nchan, pf.hdr.npol, SPECTRA_PER_ROW, 6, mads);

		//for(j = 0; j < sf.hdr.nchan * pf.hdr.npol; j++) pf.sub.dat_scales[j] = 1.0;
		for(j = 0; j < pf.hdr.nchan * pf.hdr.npol; j++) pf.sub.dat_scales[j] = (1.4826 * mads[j] * 12)/256;

		for(j = 0; j < pf.hdr.nchan * pf.hdr.npol; j++) pf.sub.dat_weights[j] = 1.0;

		/* set weights for the fake pulsar */
		//for(j = 0; j < pf.hdr.nchan * pf.hdr.npol; j++) pf.sub.dat_weights[j] = 0.0;
		//for(j = 5; j < 125; j++) pf.sub.dat_weights[j] = 1.0;
		
			
		for(j = 0; j < pf.hdr.nchan * pf.hdr.npol; j++) pf.sub.dat_offsets[j] = medians[j];
		for(j = 0; j < pf.hdr.nchan; j++) pf.sub.dat_freqs[j] = (pf.hdr.fctr - (pf.hdr.df * (pf.hdr.nchan / 2)) + ((double) j * pf.hdr.df));		
		
		fprintf(stderr, "writing subint\n");
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
	
}


	fprintf(stderr, "%d %f integnum: %d accumid: %d %f sttspec: %d stpspec: %d az: %f chan_bw: %f, exposure: %f\n", status, sf.hdr.chan_bw, sf.data_columns.integ_num, sf.data_columns.accumid, sf.data_columns.centre_freq[0], sf.data_columns.sttspec, sf.data_columns.stpspec, sf.data_columns.azimuth, sf.hdr.chan_bw, sf.data_columns.exposure);

	fprintf(stderr, "object: %s\n", name[0]);
	fprintf(stderr, "\n");
	



	/*
	float specsum[2048]; 
	for(j = 0;j < (sf.hdr.nchan * 2);j=j+1) {
		specsum[j] = 0;
	}
	
	
	for(i=0;i<sf.hdr.nrows;i++){
		 for(j = 0;j < (sf.hdr.nchan * 2);j=j+1) {		
			 specsum[j] = specsum[j] + subintdata[(2048 * i) + j];
		 }
	}	

	
		for(j = 0;j < (sf.hdr.nchan);j=j+1) {		
			 printf("%f %f\n", freqs[j], medians[j]);
		 }
	
		*/
	
	
	
	free(subintdata);

	/* close sdfits input */
	fits_close_file(sf.fptr, &status);

    
    /* flush and close psrfits output */
    fits_flush_file(pf.fptr, &status);
	fits_close_file(pf.fptr, &status);
	 
	 exit(0);
	 
	 /*
	 for (ihdu=2; ihdu<=nhdu; ihdu++) {
	   fits_movabs_hdu(fptr, ihdu, &hdutype, status);
	   if (hdutype == BINARY_TBL) {
		 fits_read_key(fptr, TSTRING, "EXTNAME", extname, comment, status);
		 fits_read_key(fptr, TSTRING, "ARRNAME", name, comment, status);
		 if (*status) {
	   *status = 0;
	   continue; 
		 }
		 if (strcmp(extname, "OI_ARRAY") != 0 || strcmp(name, arrname) != 0)
	   continue; 
	   }
	   break; 
	 }
	 */
	
	
	//fseek(fil, gethlength(buf), SEEK_CUR);
	//rv=fread(sf.sub.data, sizeof(char), sf.sub.bytes_per_subint, fil);		 
		 	
	
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



int sdfits_to_psrfits_create(struct sdfits *sf, struct psrfits *pf) {
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

    // Figure out what mode this is 

/*
    // Initialize the key variables if needed
    if (sf->new_file == 1) {  // first time writing to the file
        sf->status = 0;
        sf->tot_rows = 0;
        sf->N = 0L;
        sf->T = 0.0;
        sf->mode = 'w';

        // Create the output directory if needed
        char datadir[1024];
        strncpy(datadir, sf->basefilename, 1023);
        char *last_slash = strrchr(datadir, '/');
        if (last_slash!=NULL && last_slash!=datadir) {
            *last_slash = '\0';
            printf("Using directory '%s' for output.\n", datadir);
            char cmd[1024];
            sprintf(cmd, "mkdir -m 1777 -p %s", datadir);
            system(cmd);
        }
   		sf->new_file = 0;
    }
    sf->filenum++;
    sf->rownum = 1;

*/

	// Create basic FITS file from our template
    // Fold mode template has additional tables (polyco, ephem)
    

	
	
	char tempfilname[200];

	if(strchr(sf->filename, '/') != NULL) {
		 strncpy(pf->filename, sf->filename, strlen(sf->filename) - strlen(strrchr(sf->filename,'/')));
		 pf->filename[strlen(sf->filename) - strlen(strrchr(sf->filename,'/'))]='\0';
		 sprintf(tempfilname, "/psr_%s",strrchr(sf->filename,'/')+1);
		 strcat(pf->filename, tempfilname);	
	} else {
		 sprintf(tempfilname, "psr_%s",sf->filename);
		 strcpy(pf->filename, tempfilname);
	}

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

	fflush(stdout);
    // Check to see if file was successfully created
    if (*status) {
        fprintf(stderr, "Error creating psrfits file from template file.\n");
        fits_report_error(stderr, *status);
        exit(1);
    }



	pf->hdr.df = hdr->chan_bw/1000000;
	
	if(hdr->obsfreq < 100000) hdr->obsfreq = hdr->obsfreq * 1000000;

	if(CHANEND != 0 && CHANSTART != 0) {
		//+ (((double) (CHANEND - CHANSTART + 1) * hdr->chan_bw) ) /(1000000 * 2))
		/* new center frequency */
		pf->hdr.fctr = (hdr->obsfreq - (hdr->chan_bw * (hdr->nchan / 2)) + ((double) CHANSTART * hdr->chan_bw) ) /1000000  - (hdr->chan_bw/(2 * 1000000)) ;

		/* new observation bandwidth */
	    pf->hdr.BW = fabs(hdr->chan_bw/1000000 * (double) (CHANEND - CHANSTART + 1));

	    pf->hdr.fctr = pf->hdr.fctr - pf->hdr.BW/2; 

	} else {
	
		/* take these values from the sdfits header */
    	pf->hdr.BW = sf->hdr.efsampfr / (2 * 1000000);
		pf->hdr.fctr = hdr->obsfreq/1000000;
	
	}


	
	printf("sdfits center freq: %f psrfits center freq %f bandwidth %f...\n", hdr->obsfreq, pf->hdr.fctr, pf->hdr.BW);

	printf("hacking npol to 2...\n");
	hdr->npol = 2;
	
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

	dtmp = (double) dcols->bmaj;
    fits_update_key(pf->fptr, TDOUBLE, "BMAJ", &(dtmp), NULL, status);
	dtmp = (double) dcols->bmaj;
    fits_update_key(pf->fptr, TDOUBLE, "BMIN", &(dtmp), NULL, status);

    if (strcmp("OFF", hdr->cal_mode)) {
        fits_update_key(pf->fptr, TDOUBLE, "CAL_FREQ", &(hdr->cal_freq), NULL, status);
        fits_update_key(pf->fptr, TDOUBLE, "CAL_DCYC", &(hdr->cal_dcyc), NULL, status);
        fits_update_key(pf->fptr, TDOUBLE, "CAL_PHS", &(hdr->cal_phs), NULL, status);
    }
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
	pf->sub.offs = (((double) (dcols->integ_num + 1 - hdr->nrows)) * hdr->hwexposr) + (0.5 * hdr->hwexposr);
	
    
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
