/* quantization.c
 * Routines for requantizing 8-bit baseband data to 2-bit,
 * written by A. Siemion, 2011/03.
 */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <ctype.h>
#include <unistd.h>
#include "fitsio.h"
#include "psrfits.h"
#include "guppi_params.h"
#include "fitshead.h"

#include "quantization.h"

/* optimized 2-bit quantization */

/* applies 2 bit quantization to the data pointed to by pf->sub.data			*/
/* mean and std should be formatted as returned by 'compute_stat'				*/
/* quantization is performed 'in-place,' overwriting existing contents				*/
/* pf->hdr.nbits and pf->sub.bytes_per_subint are updated to reflect changes		*/
/* quantization scheme described at http://seti.berkeley.edu/kepler_seti_quantization  	*/

inline int quantize_2bit(struct psrfits *pf, double * mean, double * std) {

register unsigned int x,y;
unsigned int bytesperchan;


/* temporary variables for quantization routine */
float nthr[2];
float n_thr[2];
float chan_mean[2];
float sample;

register unsigned int offset;
register unsigned int address;

unsigned int pol0lookup[256];   /* Lookup tables for pols 0 and 1 */
unsigned int pol1lookup[256];


bytesperchan = pf->sub.bytes_per_subint/pf->hdr.nchan;

for(x=0;x < pf->hdr.nchan; x = x + 1)   {

		
		nthr[0] = (float) 0.98159883 * std[(x*pf->hdr.rcvr_polns) + 0];
		n_thr[0] = (float) -0.98159883 * std[(x*pf->hdr.rcvr_polns) + 0];
		chan_mean[0] = (float) mean[(x*pf->hdr.rcvr_polns) + 0];
		
		if(pf->hdr.rcvr_polns == 2) {
		   nthr[1] = (float) 0.98159883 * std[(x*pf->hdr.rcvr_polns) + 1];
		   n_thr[1] = (float) -0.98159883 * std[(x*pf->hdr.rcvr_polns) + 1];
		   chan_mean[1] = (float) mean[(x*pf->hdr.rcvr_polns) + 1];
		} else {
			nthr[1] = nthr[0];
			n_thr[1] = n_thr[0];
			chan_mean[1] = chan_mean[0];						
		}
								
		
		
		/* build the lookup table */
		for(y=0;y<128;y++) {   
			sample = ((float) y) - chan_mean[0]; 
			if (sample > nthr[0]) {
				pol0lookup[y] = 0;  						
			} else if (sample > 0) {
				pol0lookup[y] = 1; 												
			} else if (sample > n_thr[0]) {
				pol0lookup[y] = 2;																		 
			} else {
				pol0lookup[y] = 3;																		
			}	
		
			sample = ((float) y) - chan_mean[1]; 
			if (sample > nthr[1]) {
				pol1lookup[y] = 0;  						
			} else if (sample > 0) {
				pol1lookup[y] = 1; 												
			} else if (sample > n_thr[1]) {
				pol1lookup[y] = 2;																		 
			} else {
				pol1lookup[y] = 3;																		
			}			
		}
		
		for(y=128;y<256;y++) {   
			sample = ((float) y) - chan_mean[0] - 256; 
			if (sample > nthr[0]) {
				pol0lookup[y] = 0;  						
			} else if (sample > 0) {
				pol0lookup[y] = 1; 												
			} else if (sample > n_thr[0]) {
				pol0lookup[y] = 2;																		 
			} else {
				pol0lookup[y] = 3;																		
			}	
		
			sample = ((float) y) - chan_mean[1] - 256; 
			if (sample > nthr[1]) {
				pol1lookup[y] = 0;  						
			} else if (sample > 0) {
				pol1lookup[y] = 1; 												
			} else if (sample > n_thr[1]) {
				pol1lookup[y] = 2;																		 
			} else {
				pol1lookup[y] = 3;																		
			}			
		}


		/* memory position offset for this channel */
		offset = x * bytesperchan; 
		
		/* starting point for writing quantized data */
		address = offset/4;

		/* in this quantization code we'll sort-of assume that we always have two pols, but we'll set the pol0 thresholds to the pol1 values above if  */
		/* if only one pol is present. */
							
		for(y=0;y < bytesperchan; y = y + 4){
		
			/* form one 4-sample quantized byte */
			pf->sub.data[address] = pol0lookup[pf->sub.data[((offset) + y)]] + (pol0lookup[pf->sub.data[((offset) + y) + 1]] * 4) + (pol1lookup[pf->sub.data[((offset) + y) + 2]] * 16) + (pol1lookup[pf->sub.data[((offset) + y) + 3]] * 64);

			address++;																
		
		}					

}


/* update pf struct */
pf->sub.bytes_per_subint = pf->sub.bytes_per_subint / 4;			
pf->hdr.nbits = 2;			


return 1;
}


/* calculates the mean and sample std dev of the data pointed to by pf->sub.data 	*/

/* mean[0] = mean(chan 0, pol 0)	*/
/* mean[1] = mean(chan 0, pol 1)	*/
/* mean[n-1] = mean(chan n/2, pol 0)	*/
/* mean[n] = mean(chan n/2, pol 1)	*/
/* std same as above			*/


int compute_stat(struct psrfits *pf, double *mean, double *std){


double running_sum;
double running_sum_sq;
int x,y,z;
int sample;
	
 /* calulcate mean and rms for each channel-polarization */
 /* we'll treat the real and imaginary parts identically - considering them as 2 samples/period) */

/* This code is much slower than it needs to be, but it doesn't run very often */

 for(x=0;x < pf->hdr.nchan; x = x + 1)   {
		for(y=0;y<pf->hdr.rcvr_polns;y=y+1) {
			 running_sum = 0;
			 running_sum_sq = 0;
			 
			 for(z=0;z < pf->sub.bytes_per_subint/pf->hdr.nchan; z = z + (pf->hdr.rcvr_polns * 2)){
				 //pol 0, real imag

				 //real
				 sample = (int) ((signed char) pf->sub.data[(x * pf->sub.bytes_per_subint/pf->hdr.nchan) + z + (y * 2)]);
				 running_sum = running_sum + (double) sample;

				 //imag
				 sample = (int) ((signed char) pf->sub.data[(x * pf->sub.bytes_per_subint/pf->hdr.nchan) + z + (y * 2) + 1]);
				 running_sum = running_sum + (double) sample;

			 }

			 mean[(x*pf->hdr.rcvr_polns) + y] =  running_sum / (double) (pf->sub.bytes_per_subint/(pf->hdr.nchan * pf->hdr.rcvr_polns) );

			 for(z=0;z < pf->sub.bytes_per_subint/pf->hdr.nchan; z = z + (pf->hdr.rcvr_polns * 2)){
					 //sample = (int) ((signed char) pf.sub.data[(x * pf.sub.bytes_per_subint/pf.hdr.nchan) + z]);

					 //real
					 sample = (int) ((signed char) pf->sub.data[(x * pf->sub.bytes_per_subint/pf->hdr.nchan) + z + (y * 2)]);
					 running_sum_sq = running_sum_sq + pow( ((double) sample - mean[(x*pf->hdr.rcvr_polns) + y]) , 2);

					 //imag
					 sample = (int) ((signed char) pf->sub.data[(x * pf->sub.bytes_per_subint/pf->hdr.nchan) + z + (y * 2) + 1]);
					 running_sum_sq = running_sum_sq + pow( ((double) sample - mean[(x*pf->hdr.rcvr_polns) + y]) , 2);

			 }

			 std[(x*pf->hdr.rcvr_polns) + y] = pow(running_sum_sq / ((double) (pf->sub.bytes_per_subint/(pf->hdr.nchan*pf->hdr.rcvr_polns)) - 1), 0.5);
									 
			 
		}			
 }
 
return 1;

}

