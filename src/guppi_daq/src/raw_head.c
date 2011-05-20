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
double round(double x);

void print_usage(char *argv[]) {
	fprintf(stderr, "USAGE: %s -i input.raw -o output.head ('-o stdout' allowed for output, -v or -V for verbose)\n", argv[0]);
}



                                     
int main(int argc, char *argv[]) {
	struct guppi_params gf;
    struct psrfits pf;
    char buf[32768];
    char partfilename[250]; //file name for first part of file
    
	int filepos=0;
	size_t rv=0;
	int by=0;
    
    FILE *fil = NULL;   //input file
    FILE *partfil = NULL;  //partial file
    
	int x,y,z;
	int a,b,c;
	
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
			 sprintf(partfilename, optarg);
			 if(strcmp(partfilename, "stdout")==0) {
				 partfil = stdout;
			 } else {
				 partfil = fopen(partfilename, "wb");			
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




	


	if(!fil || !partfil) {
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
	 
		 //printf("%d\n", pf.hdr.nchan);    
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
					 
					 
					 fwrite(buf, sizeof(char), gethlength(buf), partfil);  //write header
					 fwrite(pf.sub.data, sizeof(char), pf.sub.bytes_per_subint, partfil);  //write data					 
					 fclose(partfil);
					 fclose(fil);
  					 exit(0);

 			 }
									
			 filepos++;

		} else {
				if(vflag>=1) fprintf(stderr, "only read %ld bytes...\n", (long int) rv);
		}

	}
		if(vflag>=1) fprintf(stderr, "bytes: %d\n",by);
		if(vflag>=1) fprintf(stderr, "pos: %ld %d\n", ftell(fil),feof(fil));
	
	
	fclose(fil);
    exit(1);
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
