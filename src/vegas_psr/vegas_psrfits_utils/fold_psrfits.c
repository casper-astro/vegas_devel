/* fold_psrfits.c
 *
 * Fold PSRFITS search data into PSRFITS folded format.
 */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <getopt.h>
#include <signal.h>
#include <pthread.h>
#include "polyco.h"
#include "fold.h"
#include "psrfits.h"

/* Signal handler */
int run=1;
void cc(int sig) { run=0; }

void usage() {
    printf(
            "Usage: fold_psrfits [options] input_filename_base\n"
            "Options:\n"
            "  -h, --help               Print this\n"
            "  -o name, --output=name   Output base filename (auto-generate)\n"
            "  -b nn, --nbin=nn         Number of profile bins (256)\n"
            "  -t nn, --tsub=n          Folded subintegration time, sec (60)\n"
            "  -j nn, --nthread=nn      Max number of threads (4)\n"
            "  -i nn, --initial=nn      Starting input file number (1)\n"
            "  -f nn, --final=nn        Ending input file number (auto)\n"
            "  -s src, --src=src        Override source name from file\n"
            "  -p file, --polyco=file   Polyco file to use (polyco.dat)\n"
            "  -P file, --parfile=file  Use given .par file\n"
            "  -F nn, --foldfreq=nn     Fold at constant freq (Hz)\n"
            "  -C, --cal                Cal folding mode\n"
            "  -u, --unsigned           Raw data is unsigned\n"
            "  -U n, --nunsigned        Num of unsigned polns\n"
            "  -S size, --split=size    Approximate max size per output file, GB (1)\n"
            "  -A, --apply              Apply input scale/offset to the results\n",
            "  -q, --quiet              No progress indicator\n"
          );
}

int main(int argc, char *argv[]) {

    /* Cmd line */
    static struct option long_opts[] = {
        {"output",  1, NULL, 'o'},
        {"nbin",    1, NULL, 'b'},
        {"tsub",    1, NULL, 't'},
        {"nthread", 1, NULL, 'j'},
        {"initial", 1, NULL, 'i'},
        {"final",   1, NULL, 'f'},
        {"src",     1, NULL, 's'},
        {"polyco",  1, NULL, 'p'},
        {"parfile", 1, NULL, 'P'},
        {"foldfreq",1, NULL, 'F'},
        {"cal",     0, NULL, 'C'},
        {"unsigned",0, NULL, 'u'},
        {"nunsigned",1, NULL, 'U'},
        {"split",   1, NULL, 'S'},
        {"apply",   0, NULL, 'A'},
        {"quiet",   0, NULL, 'q'},
        {"help",    0, NULL, 'h'},
        {0,0,0,0}
    };
    int opt, opti;
    int nbin=256, nthread=4, fnum_start=1, fnum_end=0;
    int quiet=0, raw_signed=1, use_polycos=1, cal=0, apply_scale=0;
    double split_size_gb = 1.0;
    double tfold = 60.0; 
    double fold_frequency=0.0;
    char output_base[256] = "";
    char polyco_file[256] = "";
    char par_file[256] = "";
    char source[24];  source[0]='\0';
    while ((opt=getopt_long(argc,argv,"o:b:t:j:i:f:s:p:P:F:CuU:S:Aqh",long_opts,&opti))!=-1) {
        switch (opt) {
            case 'o':
                strncpy(output_base, optarg, 255);
                output_base[255]='\0';
                break;
            case 'b':
                nbin = atoi(optarg);
                break;
            case 't':
                tfold = atof(optarg);
                break;
            case 'j':
                nthread = atoi(optarg);
                break;
            case 'i':
                fnum_start = atoi(optarg);
                break;
            case 'f':
                fnum_end = atoi(optarg);
                break;
            case 's':
                strncpy(source, optarg, 24);
                source[23]='\0';
                break;
            case 'p':
                strncpy(polyco_file, optarg, 255);
                polyco_file[255]='\0';
                use_polycos = 1;
                break;
            case 'P':
                strncpy(par_file, optarg, 255);
                par_file[255] = '\0';
                break;
            case 'F':
                fold_frequency = atof(optarg);
                use_polycos = 0;
                break;
            case 'C':
                cal = 1;
                use_polycos = 0;
                break;
            case 'u':
                raw_signed=0;
                break;
            case 'U':
                raw_signed = 4 - atoi(optarg);
                break;
            case 'S':
                split_size_gb = atof(optarg);
                break;
            case 'A':
                apply_scale = 1;
                break;
            case 'q':
                quiet=1;
                break;
            case 'h':
            default:
                usage();
                exit(0);
                break;
        }

    }
    if (optind==argc) { 
        usage();
        exit(1);
    }

    /* If no polyco/par file given, default to polyco.dat */
    if (use_polycos && (par_file[0]=='\0' && polyco_file[0]=='\0'))
        sprintf(polyco_file, "polyco.dat");

    /* Open first file */
    struct psrfits pf;
    sprintf(pf.basefilename, argv[optind]);
    pf.filenum = fnum_start;
    pf.tot_rows = pf.N = pf.T = pf.status = 0;
    pf.hdr.chan_dm = 0.0; // What if folding data that has been partially de-dispersed?
    pf.filename[0]='\0';
    int rv = psrfits_open(&pf);
    if (rv) { fits_report_error(stderr, rv); exit(1); }

    /* Check any constraints */
    if (pf.hdr.nbits!=8) { 
        fprintf(stderr, "Only implemented for 8-bit data (read nbits=%d).\n",
                pf.hdr.nbits);
        exit(1);
    }

    /* Check for calfreq */
    if (cal) {
        if (pf.hdr.cal_freq==0.0) {
            if (fold_frequency==0.0) {
                fprintf(stderr, "Error: Cal mode selected, but CAL_FREQ=0.  "
                        "Set cal frequency with -F\n");
                exit(1);
            } else {
                pf.hdr.cal_freq = fold_frequency;
            }
        } else {
            fold_frequency = pf.hdr.cal_freq;
        }
    }

    /* Set up output file */
    struct psrfits pf_out;
    memcpy(&pf_out, &pf, sizeof(struct psrfits));
    if (source[0]!='\0') { strncpy(pf_out.hdr.source, source, 24); }
    else { strncpy(source, pf.hdr.source, 24); source[23]='\0'; }
    if (output_base[0]=='\0') {
        /* Set up default output filename */
        sprintf(output_base, "%s_%s_%5.5d_%5.5d%s", pf_out.hdr.backend, 
                pf_out.hdr.source, pf_out.hdr.start_day, 
                (int)pf_out.hdr.start_sec, cal ? "_cal" : "");
    }
    sprintf(pf_out.basefilename, output_base);
    if (cal) {
        sprintf(pf_out.hdr.obs_mode, "CAL");
        sprintf(pf_out.hdr.cal_mode, "SYNC");
    } else
        sprintf(pf_out.hdr.obs_mode, "PSR");
    strncpy(pf_out.fold.parfile,par_file,255); pf_out.fold.parfile[255]='\0';
    pf_out.fptr = NULL;
    pf_out.filenum=0;
    pf_out.status=0;
    pf_out.quiet=0;
    pf_out.hdr.nbin=nbin;
    pf_out.sub.FITS_typecode = TFLOAT;
    pf_out.sub.bytes_per_subint = sizeof(float) * 
        pf_out.hdr.nchan * pf_out.hdr.npol * pf_out.hdr.nbin;
    if (split_size_gb > 0.0) { 
        pf_out.multifile = 1;
        pf_out.rows_per_file = (int) (split_size_gb * (1024.0*1024.0*1024.0)
                / (double)pf_out.sub.bytes_per_subint);
        printf("Writing a maximum of %d subintegrations (~%.1f GB) per output file.\n", 
            pf_out.rows_per_file, split_size_gb);
    } else {
        pf_out.multifile = 0;
        printf("Writing a single output file.\n");
    }

    rv = psrfits_create(&pf_out);
    if (rv) { fits_report_error(stderr, rv); exit(1); }

    /* Alloc data buffers */
    pf.sub.dat_freqs = (float *)malloc(sizeof(float) * pf.hdr.nchan);
    pf_out.sub.dat_freqs = pf.sub.dat_freqs;
    pf.sub.dat_weights = (float *)malloc(sizeof(float) * pf.hdr.nchan);
    pf_out.sub.dat_weights = (float *)malloc(sizeof(float) * pf.hdr.nchan);
    pf.sub.dat_offsets = (float *)malloc(sizeof(float) 
            * pf.hdr.nchan * pf.hdr.npol);
    pf_out.sub.dat_offsets = (float *)malloc(sizeof(float) 
            * pf.hdr.nchan * pf.hdr.npol);
    pf.sub.dat_scales  = (float *)malloc(sizeof(float) 
            * pf.hdr.nchan * pf.hdr.npol);
    pf_out.sub.dat_scales  = (float *)malloc(sizeof(float) 
            * pf.hdr.nchan * pf.hdr.npol);
    pf_out.sub.data  = (unsigned char *)malloc(pf_out.sub.bytes_per_subint);

    /* Output scale/offset */
    int i, ipol, ichan;
    float offset_uv=0.0;  
    // Extra cross-term offset for GUPPI
    if (strcmp("GUPPI",pf.hdr.backend)==0 && apply_scale==0) { 
        offset_uv=0.5;
        fprintf(stderr, "Found backend=GUPPI, setting offset_uv=%f\n",
                offset_uv);
    }
    // Initialize scale/output and weights.
    // These get copied from the input file later during the main loop.
    for (ipol=0; ipol<pf.hdr.npol; ipol++) {
        for (ichan=0; ichan<pf.hdr.nchan; ichan++) {
            float offs = 0.0;
            if (ipol>1) offs = offset_uv;
            pf_out.sub.dat_scales[ipol*pf.hdr.nchan + ichan] = 1.0;
            pf_out.sub.dat_offsets[ipol*pf.hdr.nchan + ichan] = offs;
        }
    }
    for (i=0; i<pf.hdr.nchan; i++) { pf_out.sub.dat_weights[i]=1.0; }

    /* Read or make polycos */
    int npc=0, ipc=0;
    struct polyco *pc = NULL;
    if (use_polycos) {
        if (polyco_file[0]=='\0') {
            /* Generate from par file */
            npc = make_polycos(par_file, &pf.hdr, source, &pc);
            if (npc<=0) {
                fprintf(stderr, "Error generating polycos.\n");
                exit(1);
            }
            printf("Auto-generated %d polycos, src=%s\n", npc, source);
        } else {
            /* Read from polyco file */
            FILE *pcfile = fopen(polyco_file, "r");
            if (pcfile==NULL) { 
                fprintf(stderr, "Couldn't open polyco file.\n");
                exit(1);
            }
            npc = read_all_pc(pcfile, &pc);
            if (npc==0) {
                fprintf(stderr, "Error parsing polyco file.\n");
                exit(1);
            }
            fclose(pcfile);
        }
    } else {
        // Const fold period mode, generate a fake polyco?
        pc = (struct polyco *)malloc(sizeof(struct polyco));
        sprintf(pc[0].psr, "CONST");
        pc[0].mjd = (int)pf.hdr.MJD_epoch;
        pc[0].fmjd = fmod(pf.hdr.MJD_epoch,1.0);
        pc[0].rphase = 0.0;
        pc[0].f0 = fold_frequency;
        pc[0].nsite = 0; // Does this matter?
        pc[0].nmin = 24 * 60;
        pc[0].nc = 1;
        pc[0].rf = pf.hdr.fctr;
        pc[0].c[0] = 0.0;
        pc[0].used = 0;
        npc = 1;
    }
    int *pc_written = (int *)malloc(sizeof(int) * npc);
    for (i=0; i<npc; i++) pc_written[i]=0;

    /* Alloc total fold buf */
    struct foldbuf fb;
    fb.nchan = pf.hdr.nchan;
    fb.npol = pf.hdr.npol;
    fb.nbin = pf_out.hdr.nbin;
    malloc_foldbuf(&fb);
    clear_foldbuf(&fb);

    /* Set up thread management */
    pthread_t *thread_id;
    struct fold_args *fargs;
    thread_id = (pthread_t *)malloc(sizeof(pthread_t) * nthread);
    fargs = (struct fold_args *)malloc(sizeof(struct fold_args) * nthread);
    for (i=0; i<nthread; i++) { 
        thread_id[i] = 0; 
        fargs[i].data = (char *)malloc(sizeof(char)*pf.sub.bytes_per_subint);
        fargs[i].fb = (struct foldbuf *)malloc(sizeof(struct foldbuf));
        fargs[i].fb->nbin = pf_out.hdr.nbin;
        fargs[i].fb->nchan = pf.hdr.nchan;
        fargs[i].fb->npol = pf.hdr.npol;
        fargs[i].nsamp = pf.hdr.nsblk;
        fargs[i].tsamp = pf.hdr.dt;
        fargs[i].raw_signed=raw_signed;
        malloc_foldbuf(fargs[i].fb);
        clear_foldbuf(fargs[i].fb);
        fargs[i].scale = (float *)malloc(sizeof(float) 
                * pf.hdr.nchan * pf.hdr.npol);
        fargs[i].offset = (float *)malloc(sizeof(float) 
                * pf.hdr.nchan * pf.hdr.npol);
    }

    /* Main loop */
    rv=0;
    int imjd;
    double fmjd, fmjd0=0, fmjd_next=0, fmjd_epoch;
    double offs0=0, offs1=0;
    //double phase=0.0, freq=1.0;
    int first=1, subcount=0;
    int cur_thread = 0;
    signal(SIGINT, cc);
    while (run) { 

        /* Read data block */
        pf.sub.data = (unsigned char *)fargs[cur_thread].data;
        rv = psrfits_read_subint(&pf);
        if (rv) { 
            if (rv==FILE_NOT_OPENED) rv=0; // Don't complain on file not found
            run=0; 
            break; 
        }

        /* If we've passed final file, exit */
        if (fnum_end && pf.filenum>fnum_end) { run=0; break; }

        /* Get start date, etc */
        imjd = (int)pf.hdr.MJD_epoch;
        fmjd = (double)(pf.hdr.MJD_epoch - (long double)imjd);
        fmjd += (pf.sub.offs-0.5*pf.sub.tsubint)/86400.0;

        /* First time stuff */
        if (first) {
            fmjd0 = fmjd;
            fmjd_next = fmjd + tfold/86400.0;
            pf_out.sub.offs=0.0;
            offs0 = pf.sub.offs - 0.5*pf.sub.tsubint;
            offs1 = pf.sub.offs + 0.5*pf.sub.tsubint;
            first=0;
            for (i=0; i<pf.hdr.nchan; i++) { 
                pf_out.sub.dat_weights[i]=pf.sub.dat_weights[i];
            }
        }

        /* Keep track of timestamp */
        // TODO also pointing stuff.
        pf_out.sub.offs += pf.sub.offs;
        subcount++;

        /* Update output block end time */
        offs1 = pf.sub.offs + 0.5*pf.sub.tsubint;

        /* Select polyco set */
        if (use_polycos) {
            ipc = select_pc(pc, npc, source, imjd, fmjd);
            //ipc = select_pc(pc, npc, NULL, imjd, fmjd);
            if (ipc<0) { 
                fprintf(stderr, "No matching polycos (src=%s, imjd=%d, fmjd=%f)\n",
                        source, imjd, fmjd);
                break;
            }
        } else {
            ipc = 0;
        }
        pc[ipc].used = 1; // Mark this polyco set as used for folding

        /* Copy scale/offset from input to output if we're not applying it */
        if (apply_scale==0) {
            for (i=0; i<pf.hdr.nchan*pf.hdr.npol; i++) {
                pf_out.sub.dat_scales[i] = pf.sub.dat_scales[i];
                pf_out.sub.dat_offsets[i] = pf.sub.dat_offsets[i];
            }
        }

        /* Fold this subint */
        fargs[cur_thread].pc = &pc[ipc];
        fargs[cur_thread].imjd = imjd;
        fargs[cur_thread].fmjd = fmjd;
        rv = pthread_create(&thread_id[cur_thread], NULL, 
                fold_8bit_power_thread, &fargs[cur_thread]);
        if (rv) {
            fprintf(stderr, "Thread creation error.\n");
            exit(1);
        }
        if (apply_scale) {
            for (i=0; i<pf.hdr.nchan*pf.hdr.npol; i++) {
                fargs[cur_thread].scale[i] = pf.sub.dat_scales[i];
                fargs[cur_thread].offset[i] = pf.sub.dat_offsets[i];
            }
        }
        cur_thread++;

        /* Combine thread results if needed */
        if (cur_thread==nthread || fmjd>fmjd_next) {

            /* Loop over active threads */
            for (i=0; i<cur_thread; i++) {

                /* Wait for thread to finish */
                rv = pthread_join(thread_id[i], NULL);
                if (rv) { 
                    fprintf(stderr, "Thread join error.\n");
                    exit(1);
                }

                /* Apply scale and offset here */
                if (apply_scale) 
                    scale_offset_folds(fargs[i].fb, fargs[i].scale,
                            fargs[i].offset);

                /* Combine its result into total fold */
                accumulate_folds(&fb, fargs[i].fb);

                /* Reset thread info */
                clear_foldbuf(fargs[i].fb);
                thread_id[i] = 0;

            }

            /* Reset active thread count */
            cur_thread = 0;
        }

        /* See if integration needs to be written, etc */
        if (fmjd > fmjd_next) {

            /* Figure out timestamp */
            pf_out.sub.offs /= (double)subcount;
            pf_out.sub.tsubint = offs1 - offs0;
            fmjd_epoch = fmjd0 + pf_out.sub.offs/86400.0;
            /*
            // Don't need this stuff if we set EPOCHS=MIDTIME
            ipc = select_pc(pc, npc, pf.hdr.source, imjd, fmjd_epoch); 
            if (ipc<0) { 
                fprintf(stderr, "Polyco error, exiting.\n");
                exit(1);
            }
            phase = psr_phase(&pc[ipc], imjd, fmjd_epoch, &freq);
            phase = fmod(phase, 1.0);
            pf_out.sub.offs -= phase/freq; // ref epoch needs 0 phase
            */

            /* Transpose, output subint */
            normalize_transpose_folds((float *)pf_out.sub.data, &fb);
            int last_filenum = pf_out.filenum;
            psrfits_write_subint(&pf_out);


            /* Check for write errors */
            if (pf_out.status) {
                fprintf(stderr, "Error writing subint.\n");
                fits_report_error(stderr, pf_out.status);
                exit(1);
            }

            /* Check if we started a new file */
            if (pf_out.filenum!=last_filenum) {
                /* No polycos yet written to this file */
                for (i=0; i<npc; i++) pc_written[i]=0;
            }

            /* Write the current polyco if needed */
            if (pc_written[ipc]==0) {
                psrfits_write_polycos(&pf_out, &pc[ipc], 1);
                if (pf_out.status) {
                    fprintf(stderr, "Error writing polycos.\n");
                    fits_report_error(stderr, pf_out.status);
                    exit(1);
                }
                pc_written[ipc] = 1;
            }

            /* Clear counters, avgs */
            clear_foldbuf(&fb);
            pf_out.sub.offs = 0.0;
            offs0 = pf.sub.offs - 0.5*pf.sub.tsubint;
            subcount=0;

            /* Set next output time */
            fmjd_next = fmjd + tfold/86400.0;
        }


        /* Progress report */
        if (!quiet) {
            printf("\rFile %d %5.1f%%", pf.filenum, 
                    100.0 * (float)(pf.rownum-1)/(float)pf.rows_per_file);
            fflush(stdout);
        }
    }

    /* Join any running threads */
    for (i=0; i<cur_thread; i++)  
        if (thread_id[i]) pthread_join(thread_id[i], NULL);

    /* Remove polyco table in cal mode */
    if (cal) {
        rv = psrfits_remove_polycos(&pf_out);
        if (rv) { fits_report_error(stderr, rv); }
    }

    psrfits_close(&pf_out);
    psrfits_close(&pf);

    if (rv) { fits_report_error(stderr, rv); }
    exit(0);
}
