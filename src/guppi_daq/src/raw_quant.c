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

/* Parse info from buffer into param struct */
extern void guppi_read_obs_params(char *buf, 
                                     struct guppi_params *g,
                                     struct psrfits *p);


void print_usage(char *argv[]) {
	fprintf(stderr, "USAGE: %s -i input.raw -o output.quant ('-o stdout' allowed for output, -v or -V for verbose)\n", argv[0]);
}



/* 03/13 - edit to do optimized inplace quantization */

                                     
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

	
	double *mean = NULL;
	double *std = NULL;
	
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

		 if(vflag>=1) fprintf(stderr, "length: %d\n", gethlength(buf));

		 guppi_read_obs_params(buf, &gf, &pf);
	 
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
				/* beginning of file, compute statistics */
				
			    mean = malloc(pf.hdr.rcvr_polns *  pf.hdr.nchan * sizeof(double));
			    std = malloc(pf.hdr.rcvr_polns *  pf.hdr.nchan * sizeof(double));

			   // memset(mean, 0, pf.hdr.rcvr_polns *  pf.hdr.nchan * sizeof(double));
			   // memset(std, 0, pf.hdr.rcvr_polns *  pf.hdr.nchan * sizeof(double));
	
 				compute_stat(&pf, mean, std);

//			 if(vflag>=1) fprintf(stderr, "chan  %d pol %d mean %f\n", x,y,mean[(x*pf->hdr.rcvr_polns) + y]);
//			 if(vflag>=1) fprintf(stderr, "chan  %d pol %d std %f\n", x,y,std[(x*pf->hdr.rcvr_polns) + y]);


 			 }


			quantize_2bit(&pf, mean, std);


			hputi4 (buf, "BLOCSIZE", pf.sub.bytes_per_subint);
			hputi4 (buf,"NBITS",pf.hdr.nbits);

			fwrite(buf, sizeof(char), gethlength(buf), quantfil);  //write header
			
			/* bytes_per_subint now updated to be the proper length */
			fwrite(pf.sub.data, sizeof(char), pf.sub.bytes_per_subint, quantfil);  //write data

			filepos++;

			 
		} else {
				if(vflag>=1) fprintf(stderr, "only read %ld bytes...\n", (long int) rv);
		}

	}
		if(vflag>=1) fprintf(stderr, "bytes: %d\n",by);
		if(vflag>=1) fprintf(stderr, "pos: %ld %d\n", ftell(fil),feof(fil));
	



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
