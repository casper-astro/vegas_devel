#include <stdio.h>
#include <stdlib.h>
#include "fitsio.h"
#include "psrfits.h"
#include "guppi_params.h"
#include "fitshead.h"
#include <math.h>
#include <arpa/inet.h>
#include <string.h>

/* Parse info from buffer into param struct */
extern void guppi_read_obs_params(char *buf, 
                                     struct guppi_params *g,
                                     struct psrfits *p);
double round(double x);
                       

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
	struct guppi_params gf;
    struct psrfits pf;
    char buf[32768];
    char partfilename[250]; //file name for first part of file
    char quantfilename[250]; //file name for first part of file
    char keywrd[250];
	int subintcnt=0;
	int filepos=0;
	size_t rv=0;
	int by=0;
    
    FILE *fil;   //input file
    FILE *partfil;  //partial file
    FILE *quantfil;  //quantized file
    
	int a,x,y,z;
	double power;
	int sampsper = 8192;
	int sample;
	
	double running_sum;
	double running_sum_sq;
	double mean[32];   //shouldn't be more than 32 channels in a file?
	double std[32];
    unsigned char quantbyte;
    
    
    char *fitsdata = NULL;
    
    
    float nthr;
    float n_thr;
    
    float fitsval;
    unsigned int quantval;

    if(argc < 3) {
		fprintf(stderr, "USAGE: %s input.raw output.fits (use 'stdout' for output if stdout is desired\n", argv[0]);
		exit(1);
	}
    
    float quantlookup[4];
    quantlookup[0] = 3.3358750;
    quantlookup[1] = 1.0;
    quantlookup[2] = -1.0;
    quantlookup[3] = -3.3358750;
    sprintf(pf.basefilename, argv[1]);

	sprintf(partfilename, argv[2]);

    pf.filenum=1;
    pf.sub.dat_freqs = (float *)malloc(sizeof(float) * pf.hdr.nchan);
    pf.sub.dat_weights = (float *)malloc(sizeof(float) * pf.hdr.nchan);
    pf.sub.dat_offsets = (float *)malloc(sizeof(float) 
           * pf.hdr.nchan * pf.hdr.npol);
    pf.sub.dat_scales  = (float *)malloc(sizeof(float) 
            * pf.hdr.nchan * pf.hdr.npol);
    pf.sub.data  = (unsigned char *)malloc(pf.sub.bytes_per_subint);





	
	fil = fopen(pf.basefilename, "rb");
	partfil = fopen(partfilename, "wb");	
	
	filepos=0;
	
	while(fread(buf, sizeof(char), 32768, fil)==32768) {		

		 fseek(fil, -32768, SEEK_CUR);
		 //printf("lhead: %d", lhead0);
		 fprintf(stderr, "length: %d\n", gethlength(buf));

		 guppi_read_obs_params(buf, &gf, &pf);
	 
		 //printf("%d\n", pf.hdr.nchan);    
		 fprintf(stderr, "size %d\n",pf.sub.bytes_per_subint + gethlength(buf));
		 by = by + pf.sub.bytes_per_subint + gethlength(buf);
		 fprintf(stderr, "mjd %Lf\n", pf.hdr.MJD_epoch);
		 fprintf(stderr, "zen: %f\n\n", pf.sub.tel_zen);

		 if (pf.sub.data) {
		 	printf("boo sub\n");
		 	fflush(stdout);
		 	free(pf.sub.data);         
         }
         pf.sub.data  = (unsigned char *)malloc(pf.sub.bytes_per_subint);
		 
		 //need to allocate 4 bytes for each sample (float vals)
		 if (fitsdata) {
		 	printf("boo fits\n");
			fflush(stdout);
		 	free(fitsdata);         
		 }
		 fitsdata = (char *) malloc(pf.sub.bytes_per_subint*4* (8/pf.hdr.nbits));
		 
		 fseek(fil, gethlength(buf), SEEK_CUR);
		 rv=fread(pf.sub.data, sizeof(char), pf.sub.bytes_per_subint, fil);		 
		 


		
		 if((long int)rv == pf.sub.bytes_per_subint){
			 fprintf(stderr, "%i\n", filepos);
			 fprintf(stderr, "pos: %ld %d\n", ftell(fil),feof(fil));


			 if(filepos == 0) {
					 //first time through, save first part of file

					 //SIMPLE  =                    T / file does conform to FITS standard
					 //BITPIX  =                    8 / number of bits per data pixel
					 //NAXIS   =                    0 / number of data axes
					 
					 for(x=0;x<26;x++) {
					 	sprintf(keywrd, "FILL%d", x); 
					 	hadd(buf, keywrd);
					 }

					 hadd(buf, "NAXIS2");
					 hadd(buf, "NAXIS1");					 
					 hadd(buf, "NAXIS");
					 hadd(buf, "BITPIX");
					 hadd(buf, "SIMPLE");
					 hputc(buf, "SIMPLE", "T");
					 hputi4(buf, "BITPIX", -32);
					 hputi4(buf, "NAXIS", 2);
					 hputi4(buf, "NAXIS1", pf.sub.bytes_per_subint * (8/pf.hdr.nbits));
					 hputi4(buf, "NAXIS2", 1);
					 
					 
					 printf("NBITS IS: %d\n", pf.hdr.nbits);
					 //hputi4 (buf, "", pf.sub.bytes_per_subint/4);
					 printf("wrote: %d\n",fwrite(buf, sizeof(char), gethlength(buf), partfil));  //write header
					 z=0;


					for(x=0;x < pf.sub.bytes_per_subint ;x=x+1) {
//							printf("%d\n", (int) ((signed char) pf.sub.data[x]));
						 	//printf("blah %d\n",z);
						 	//z=x+1;


						if(pf.hdr.nbits == 2){
							 for(a=0;a<4;a++){
								 quantval=0;
								 quantval = quantval + (pf.sub.data[x] >> (a * 2) & 1);
								 quantval = quantval + (2 * (pf.sub.data[x] >> (a * 2 + 1) & 1));
																 
								 //printf("%u\n", quantval);							
								 
								 fitsval = quantlookup[quantval];
								 //printf("%f\n", fitsval);							
								 //usleep(1000000);

								 memcpy(&fitsdata[z], &fitsval, sizeof(float));						 	
							     z = z + 4;									 
								 //fprintf(stderr, "%u\n", quantval);
								 //usleep(1000000);
							 }
						} else {						
						 	fitsval = ((float) (signed char) pf.sub.data[x]) ;
						 	memcpy(&fitsdata[z], &fitsval, sizeof(float));						 	
							z = z + 4;
						}	
							//printf("%f\n", fitsval);
						 	//fitsval = (float) htonl((unsigned int) fitsval);
							//printf("%f\n", fitsval);							
					 		//usleep(1000000);
							
							//fwrite(&fitsval, sizeof(float), 1, partfil);
							
						 	//bin_print_verbose(fitsval);
						 	//fitsval = ((fitsval >> 8) & 0x00ff) | ((fitsval & 0x00ff) << 8);
						 	
					}
					
					
					
					printf("%d\n",x);
					imswap4(fitsdata,pf.sub.bytes_per_subint*4* (8/pf.hdr.nbits));
					fwrite(fitsdata, sizeof(char), pf.sub.bytes_per_subint*4* (8/pf.hdr.nbits), partfil);
					fflush(partfil);
					fclose(fil);
					fclose(partfil);
					exit(0);					 	
 			 }
		
		} else {
				fprintf(stderr, "only read %ld bytes...\n", (long int) rv);
		}

	}
		fprintf(stderr, "bytes: %d\n",by);
		fprintf(stderr, "pos: %ld %d\n", ftell(fil),feof(fil));
	
	
	fclose(fil);
    exit(0);
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
