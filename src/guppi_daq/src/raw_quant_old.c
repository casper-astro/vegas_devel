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

/* Parse info from buffer into param struct */
extern void guppi_read_obs_params(char *buf, 
                                     struct guppi_params *g,
                                     struct psrfits *p);

int quantize_2bit(struct psrfits *pf, double mean[][2], double std[][2], int vflag);
inline int quantize_2bit_o(struct psrfits *pf, double mean[][2], double std[][2]);

void print_usage(char *argv[]) {
	fprintf(stderr, "USAGE: %s -i input.raw -o output.quant ('-o stdout' allowed for output, -v or -V for verbose)\n", argv[0]);
}

/* 03/13 - edit to do inplace quantization */



                                     
int main(int argc, char *argv[]) {
	struct guppi_params gf;
    struct psrfits pf;
    char buf[32768];
    char quantfilename[250]; //file name for quantized file
    
	int filepos=0;
	size_t rv=0;
	int by=0;
    
    FILE *fil = NULL;   //input file
    FILE *quantfil = NULL;  //quantized file
    
	int x,y,z;
	int a,b,c;
	int sample;
	
	double running_sum;
	double running_sum_sq;
	double mean[32][2];   //shouldn't be more than 32 dual pol channels in a file for our obs
	double std[32][2];
	
    int vflag=0; //verbose

    

    
	   if(argc < 2) {
		   print_usage(argv);
		   exit(1);
	   }


       opterr = 0;
     
       while ((c = getopt (argc, argv, "Vvi:o:")) != -1)
         switch (c)
           {
           case 'v':
             vflag = 1;
             break;
           case 'V':
             vflag = 2;
             break; 
           case 'i':
			 sprintf(pf.basefilename, optarg);
			 fil = fopen(pf.basefilename, "rb");
             break;
           case 'o':
			 sprintf(quantfilename, optarg);
			 if(strcmp(quantfilename, "stdout")==0) {
				 quantfil = stdout;
			 } else {
				 quantfil = fopen(quantfilename, "wb");			
			 }
             break;
           case '?':
             if (optopt == 'i' || optopt == 'o')
               fprintf (stderr, "Option -%c requires an argument.\n", optopt);
             else if (isprint (optopt))
               fprintf (stderr, "Unknown option `-%c'.\n", optopt);
             else
               fprintf (stderr,
                        "Unknown option character `\\x%x'.\n",
                        optopt);
             return 1;
           default:
             abort ();
           }


   

    pf.filenum=1;
    pf.sub.dat_freqs = (float *)malloc(sizeof(float) * pf.hdr.nchan);
    pf.sub.dat_weights = (float *)malloc(sizeof(float) * pf.hdr.nchan);
    pf.sub.dat_offsets = (float *)malloc(sizeof(float) 
           * pf.hdr.nchan * pf.hdr.npol);
    pf.sub.dat_scales  = (float *)malloc(sizeof(float) 
            * pf.hdr.nchan * pf.hdr.npol);
    pf.sub.data  = (unsigned char *)malloc(pf.sub.bytes_per_subint);




	

	if(!fil || !quantfil) {
		fprintf(stderr, "must specify input/output files\n");
		print_usage(argv);
		exit(1);
	}
	
	filepos=0;
	
	while(fread(buf, sizeof(char), 32768, fil)==32768) {		

		 fseek(fil, -32768, SEEK_CUR);
		 //printf("lhead: %d", lhead0);
		 if(vflag>=1) fprintf(stderr, "length: %d\n", gethlength(buf));

		 guppi_read_obs_params(buf, &gf, &pf);
	 
		 printf("%d\n", pf.hdr.nbits);    
		 if(vflag>=1) fprintf(stderr, "size %d\n",pf.sub.bytes_per_subint + gethlength(buf));
		 by = by + pf.sub.bytes_per_subint + gethlength(buf);
		 if(vflag>=1) fprintf(stderr, "mjd %Lf\n", pf.hdr.MJD_epoch);
		 if(vflag>=1) fprintf(stderr, "zen: %f\n\n", pf.sub.tel_zen);
		 if (pf.sub.data) free(pf.sub.data);
         pf.sub.data  = (unsigned char *)malloc(pf.sub.bytes_per_subint);
		 
		 fseek(fil, gethlength(buf), SEEK_CUR);
		 rv=fread(pf.sub.data, sizeof(char), pf.sub.bytes_per_subint, fil);		 
		 


		
		 if((long int)rv == pf.sub.bytes_per_subint){
			 if(vflag>=1) fprintf(stderr, "%i\n", filepos);
			 if(vflag>=1) fprintf(stderr, "pos: %ld %d\n", ftell(fil),feof(fil));


			 if(filepos == 0) {
					 
					 /* calulcate mean and rms for each channel-polarization */
					 /* we'll treat the real and imaginary parts identically - considering them as 2 samples/period) */

					
/*
					 for(x=0;x < pf.hdr.nchan; x = x + 1)   {
							for(y=0;y<pf.hdr.rcvr_polns;y=y+1) {
								 running_sum = 0;
								 running_sum_sq = 0;
								 
								 for(z=0;z < pf.sub.bytes_per_subint/pf.hdr.nchan; z = z + (pf.hdr.rcvr_polns * 2)){
									 //pol 0, real imag

									 //real
									 sample = (int) ((signed char) pf.sub.data[(x * pf.sub.bytes_per_subint/pf.hdr.nchan) + z + (y * 2)]);
									 running_sum = running_sum + (double) sample;

									 //imag
									 sample = (int) ((signed char) pf.sub.data[(x * pf.sub.bytes_per_subint/pf.hdr.nchan) + z + (y * 2) + 1]);
									 running_sum = running_sum + (double) sample;

								 }
			 
								 mean[x][y] =  running_sum / (double) (pf.sub.bytes_per_subint/(pf.hdr.nchan * pf.hdr.rcvr_polns) );
			 
								 for(z=0;z < pf.sub.bytes_per_subint/pf.hdr.nchan; z = z + (pf.hdr.rcvr_polns * 2)){
										 //sample = (int) ((signed char) pf.sub.data[(x * pf.sub.bytes_per_subint/pf.hdr.nchan) + z]);

										 //real
										 sample = (int) ((signed char) pf.sub.data[(x * pf.sub.bytes_per_subint/pf.hdr.nchan) + z + (y * 2)]);
										 running_sum_sq = running_sum_sq + pow( ((double) sample - mean[x][y]) , 2);
	
										 //imag
										 sample = (int) ((signed char) pf.sub.data[(x * pf.sub.bytes_per_subint/pf.hdr.nchan) + z + (y * 2) + 1]);
										 running_sum_sq = running_sum_sq + pow( ((double) sample - mean[x][y]) , 2);

								 }

								 std[x][y] = pow(running_sum_sq / ((double) (pf.sub.bytes_per_subint/(pf.hdr.nchan*pf.hdr.rcvr_polns)) - 1), 0.5);
														 
								 
								 if(vflag>=1) fprintf(stderr, "chan  %d pol %d mean %f\n", x,y,mean[x][y]);
								 if(vflag>=1) fprintf(stderr, "chan  %d pol %d std %f\n", x,y,std[x][y]);
							}			
					 }
 			 		*/


 					 for(x=0;x<32;x++){
 			 		 	std[x][0] = 19;
 			 		 	mean[x][0] = 0;		
 			 		 	std[x][1] = 19;
 			 		 	mean[x][1] = 0;		
 			 		 }
 			 }


			quantize_2bit_o(&pf, mean, std);

			hputi4 (buf, "BLOCSIZE", pf.sub.bytes_per_subint);
			hputi4 (buf,"NBITS",pf.hdr.nbits);

			fwrite(buf, sizeof(char), gethlength(buf), quantfil);  //write header
			
			/* bytes_per_subint now updated to be the proper length */
			fwrite(pf.sub.data, sizeof(char), pf.sub.bytes_per_subint, quantfil);  //write data

			filepos++;

			 //pol, time, frequency
			 
//			 for(x=0; x < pf.sub.bytes_per_subint; x = x + 1) {
//			 	fprintf(stderr, "old: %d", pf.sub.data[x]);
//			 	if(pf.sub.data[x] >  127) { pf.sub.data[x] = pf.sub.data[x] - 256; 
//			 	fprintf(stderr, "new: %d", pf.sub.data[x]); }
//			 }
			 
			 /*
			 for(x=0; x < pf.sub.bytes_per_subint/pf.hdr.nchan; x = x + (4 * sampsper)) {
				power=0;
				for(z=0;z<sampsper;z=z+4){						
			 		for(y=0;y<4;y++) {
						sample = (int) pf.sub.data[x+y+(z*4)];
						if(sample > 127) sample = sample - 256;
			 			power = power + pow((double) sample, 2);
					}
				}
			 	//printf("%d, %d\n", pf.sub.data[x], pf.sub.data[x+1]); 			 
				printf("%f\n",power);			  

			  }
			  */
		} else {
				if(vflag>=1) fprintf(stderr, "only read %ld bytes...\n", (long int) rv);
		}

	}
		if(vflag>=1) fprintf(stderr, "bytes: %d\n",by);
		if(vflag>=1) fprintf(stderr, "pos: %ld %d\n", ftell(fil),feof(fil));
	
	
	//fread(buf, sizeof(char), 32768, fil);

    //guppi_read_obs_params(buf, &gf, &pf);
	//printf("mjd %llf\n", pf.hdr.MJD_epoch);
    //printf("zen: %f", pf.sub.tel_zen);


    //while ((rv=psrfits_read_subint(&pf))==0) { 
    //    printf("Read subint (file %d, row %d/%d)\n", 
    //            pf.filenum, pf.rownum-1, pf.rows_per_file);
    //}
    //if (rv) { fits_report_error(stderr, rv); }
	fclose(quantfil);
	fclose(fil);
    exit(0);
}



void bin_print_verbose(unsigned char x)
/* function to print decimal numbers in verbose binary format */
/* x is integer to print, n_bits is number of bits to print */
{

   int j;
   printf("no. 0x%08x in binary \n",(int) x);

   for(j=8-1; j>=0;j--){
	   printf("bit: %i = %i\n",j, (x>>j) & 01);
   }

}


/* non-optimized 2 bit quantization */
int quantize_2bit(struct psrfits *pf, double mean[][2], double std[][2], int vflag) {

int x,y,z;

/* temporary variables for quantization routine */
float nthr[2];
float n_thr[2];
float chan_mean[2];
float sample_fl;
unsigned char quantbyte;



			for(x=0;x < pf->hdr.nchan; x = x + 1)   {
					if(vflag>=1) fprintf(stderr, "on channel: %d\n",x);
					/* subint/nchan should always be a multiple of 4 for dual pol complex data? */
					//nthr = round(0.98159883 * std[x]);
					//n_thr = round(-0.98159883 * std[x]);
					
					
					nthr[0] = (float) 0.98159883 * std[x][0];
					n_thr[0] = (float) -0.98159883 * std[x][0];
					chan_mean[0] = (float) mean[x][0];

					if(pf->hdr.rcvr_polns == 2) {
					   nthr[1] = (float) 0.98159883 * std[x][1];
					   n_thr[1] = (float) -0.98159883 * std[x][1];
					   chan_mean[1] = (float) mean[x][1];
					} else {
						nthr[1] = (float) 0.98159883 * std[x][0];
						n_thr[1] = (float) -0.98159883 * std[x][0];
						chan_mean[1] = (float) mean[x][0];						
					}
					
					
					/* in this quantization code we'll sort-of assume that we always have two pols, but we'll set the pol2 thresholds to the pol1 values above if only one pol */
					/* this is just to make the loop below as simple as possible to make optimization as easy as possible for the compiler */
					
					for(y=0;y < pf->sub.bytes_per_subint/pf->hdr.nchan; y = y + 4){
						//pol 0, real imag
						quantbyte = 0;
					
						//bin_print_verbose(quantbyte);
//						printf("%d\n", (int) ((signed char) pf.sub.data[(x * pf.sub.bytes_per_subint/pf.hdr.nchan) + y + 0]));
						//printf("%i\n", ((signed char) pf.sub.data[(x * pf.sub.bytes_per_subint/pf.hdr.nchan) + y + 0]));
						
						//binary value  	float		unsigned int
						// 00    			n   		0
						// 10    			1   		1
						// 01   			-1 	 		2
						// 11   			-n   		3
						
						//set bits 0,1
						
						for(z=0;z<2;z++){
							sample_fl = ((float) (signed char) pf->sub.data[(x * pf->sub.bytes_per_subint/pf->hdr.nchan) + y + z]) - chan_mean[0];
							
							//fprintf(stderr, "comparing: %f %f\n", nthr, ((float) (signed char) pf.sub.data[(x * pf.sub.bytes_per_subint/pf.hdr.nchan) + y + z]));
							if (sample_fl >= nthr[0]) {
								quantbyte += (0<<((z*2)+0)); 
								quantbyte += (0<<((z*2)+1)); 						
								if(vflag>=2) fprintf(stderr," %f : 00 : 3.3358750 ", sample_fl);
							} else if (sample_fl >= 0) {
								quantbyte += (1<<((z*2)+0)); 
								quantbyte += (0<<((z*2)+1)); 												
								if(vflag>=2) fprintf(stderr," %f : 10 : 1 ", sample_fl);
							} else if (sample_fl >= n_thr[0]) {
								quantbyte += (0<<((z*2)+0)); 
								quantbyte += (1<<((z*2)+1)); 																		 
								if(vflag>=2) fprintf(stderr," %f : 01 : -1 ", sample_fl);
							} else {
								quantbyte += (1<<((z*2)+0)); 
								quantbyte += (1<<((z*2)+1)); 																		
								if(vflag>=2) fprintf(stderr," %f : 11 : -3.3358750 ", sample_fl);
							}																									
						}


						for(z=2;z<4;z++){
							sample_fl = ((float) (signed char) pf->sub.data[(x * pf->sub.bytes_per_subint/pf->hdr.nchan) + y + z]) - chan_mean[1];
							
							//fprintf(stderr, "comparing: %f %f\n", nthr, ((float) (signed char) pf.sub.data[(x * pf.sub.bytes_per_subint/pf.hdr.nchan) + y + z]));
							if (sample_fl >= nthr[1]) {
								quantbyte += (0<<((z*2)+0)); 
								quantbyte += (0<<((z*2)+1)); 						
								if(vflag>=2) fprintf(stderr," %f : 00 : 3.3358750 ", sample_fl);
							} else if (sample_fl >= 0) {
								quantbyte += (1<<((z*2)+0)); 
								quantbyte += (0<<((z*2)+1)); 												
								if(vflag>=2) fprintf(stderr," %f : 10 : 1 ", sample_fl);
							} else if (sample_fl >= n_thr[1]) {
								quantbyte += (0<<((z*2)+0)); 
								quantbyte += (1<<((z*2)+1)); 																		 
								if(vflag>=2) fprintf(stderr," %f : 01 : -1 ", sample_fl);
							} else {
								quantbyte += (1<<((z*2)+0)); 
								quantbyte += (1<<((z*2)+1)); 																		
								if(vflag>=2) fprintf(stderr," %f : 11 : -3.3358750 ", sample_fl);
							}																									
						}

						
												
						if(vflag>=2) fprintf(stderr, "\n%u\n", (unsigned char) quantbyte);
						if(vflag>=2) usleep(100000);
						
						
						//quantdata[((x * pf->sub.bytes_per_subint/pf->hdr.nchan) + y)/4] = quantbyte;
						pf->sub.data[((x * pf->sub.bytes_per_subint/pf->hdr.nchan) + y)/4] = quantbyte;

						//quantbyte += (1<<3); //& (1<<i);
						//quantbyte += (1<<1); //& (1<<i);

						//set bits 2,3
						//set bits 4,5
						//set bits 6,7


						//bin_print_verbose(quantbyte);

						//usleep(1000000);
						//((signed char) pf.sub.data[(x * pf.sub.bytes_per_subint/pf.hdr.nchan) + y + 1]);
						//((signed char) pf.sub.data[(x * pf.sub.bytes_per_subint/pf.hdr.nchan) + y + 2]);
						//((signed char) pf.sub.data[(x * pf.sub.bytes_per_subint/pf.hdr.nchan) + y + 3]);
											
					}					
	
			}

			/* update pf struct */
			pf->sub.bytes_per_subint = pf->sub.bytes_per_subint / 4;			
			pf->hdr.nbits = 2;			
			
			return 1;

}


/* optimized 2-bit quantization */
inline int quantize_2bit_o(struct psrfits *pf, double mean[][2], double std[][2]) {

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

					/* subint/nchan should always be a multiple of 4 for dual pol complex data? */
					//nthr = round(0.98159883 * std[x]);
					//n_thr = round(-0.98159883 * std[x]);
					
					
					nthr[0] = (float) 0.98159883 * std[x][0];
					n_thr[0] = (float) -0.98159883 * std[x][0];
					chan_mean[0] = (float) mean[x][0];

					if(pf->hdr.rcvr_polns == 2) {
					   nthr[1] = (float) 0.98159883 * std[x][1];
					   n_thr[1] = (float) -0.98159883 * std[x][1];
					   chan_mean[1] = (float) mean[x][1];
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
					
						pf->sub.data[address] = pol0lookup[pf->sub.data[((offset) + y)]] + (pol0lookup[pf->sub.data[((offset) + y) + 1]] * 4) + (pol1lookup[pf->sub.data[((offset) + y) + 2]] * 16) + (pol1lookup[pf->sub.data[((offset) + y) + 3]] * 64);
						address++;																
					
					}					
	
			}


			/* update pf struct */
			pf->sub.bytes_per_subint = pf->sub.bytes_per_subint / 4;			
			pf->hdr.nbits = 2;			


			return 1;

}


						//pf->sub.data[address] = pol0lookup[pf->sub.data[((offset) + y)]];
						//pf->sub.data[address] += pol0lookup[pf->sub.data[((offset) + y) + 1]] * 4;
						//pf->sub.data[address] += pol1lookup[pf->sub.data[((offset) + y) + 2]] * 16;
						//pf->sub.data[address] += pol1lookup[pf->sub.data[((offset) + y) + 3]] * 64;


							//printf("%f %d\n", sample, pol0lookup[pf->sub.data[((offset) + y)]]);
							//sample_fl2 = lookup[pf->sub.data[((offset) + y) + 1]] - chan_mean[0];
							//sample_fl3 = lookup[pf->sub.data[((offset) + y) + 2]] - chan_mean[1];
							//sample_fl4 = lookup[pf->sub.data[((offset) + y) + 3]] - chan_mean[1];
							//usleep(500000);
				
						//pol 0, real imag
							//sample_fl1 = ((float) (signed char) pf->sub.data[((offset) + y)]) - chan_mean[0];					
							//sample_fl2 = ((float) (signed char) pf->sub.data[((offset) + y) + 1]) - chan_mean[0];
							//sample_fl3 = ((float) (signed char) pf->sub.data[((offset) + y) + 2]) - chan_mean[1];
							//sample_fl4 = ((float) (signed char) pf->sub.data[((offset) + y) + 3]) - chan_mean[1];

/*
							sample_fl1 = lookup[pf->sub.data[((offset) + y)]] - chan_mean[0];
							//printf("%f\n", sample_fl1);
							sample_fl2 = lookup[pf->sub.data[((offset) + y) + 1]] - chan_mean[0];
							sample_fl3 = lookup[pf->sub.data[((offset) + y) + 2]] - chan_mean[1];
							sample_fl4 = lookup[pf->sub.data[((offset) + y) + 3]] - chan_mean[1];
							//usleep(500000);
							
							

							if (sample_fl1 > nthr[0]) {
								//pf->sub.data[address] += 0;  						
							} else if (sample_fl1 > 0) {
								pf->sub.data[address] += 1; 												
							} else if (sample_fl1 > n_thr[0]) {
								pf->sub.data[address] += 2;																		 
							} else {
								pf->sub.data[address] += 3;																		
							}																										
							
							
							if (sample_fl2 > nthr[0]) {
								//pf->sub.data[address] += 0;  						
							} else if (sample_fl2 > 0) {
								pf->sub.data[address] += 4;  												
							} else if (sample_fl2 > n_thr[0]) {
								pf->sub.data[address] += 8;																		 
							} else {
								pf->sub.data[address] += 12; 																	
							}																									
							
							
							if (sample_fl3 > nthr[1]) {
								//pf->sub.data[address] += 0; 
							} else if (sample_fl3 > 0) {
								pf->sub.data[address] += 16;  												
							} else if (sample_fl3 > n_thr[1]) {
								pf->sub.data[address] += 32; 																		 
							} else {
								pf->sub.data[address] += 48; 																		
							}		

							
							if (sample_fl4 > nthr[1]) {
								//quantbyte += 0;
							} else if (sample_fl4 > 0) {
								pf->sub.data[address] += 64;										
							} else if (sample_fl4 > n_thr[1]) {
								pf->sub.data[address] += 128;
							} else {
								pf->sub.data[address] += 192; 
							}																									
												
*/						