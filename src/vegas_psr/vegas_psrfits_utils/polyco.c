/* polyco.c
 * routines to read/use polyco.dat
 */

#include "polyco.h"
#include "psrfits.h"
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>
#include <math.h>

int read_one_pc(FILE *f, struct polyco *pc) {

    int i, j;
    char *rv;
    int ret;
    char buf[90];
    /* Read in polyco chunk */
    rv = fgets(buf, 90, f);
    if (rv==NULL) { return(-1); }
    strncpy(pc->psr, &buf[0], 10);  pc->psr[10] = '\0';
    pc->mjd = atoi(&buf[31]);
    pc->fmjd = atof(&buf[39]);
    if ((rv=strchr(pc->psr, ' '))!=NULL) { *rv='\0'; }
    rv = fgets(buf,90,f);
    if (rv==NULL) { return(-1); }
    pc->rphase_int = atoll(&buf[0]);
    pc->rphase = fmod(atof(&buf[0]),1.0);
    pc->f0 = atof(&buf[20]);
    pc->nsite = atoi(&buf[42]);
    pc->nmin = atoi(&buf[43]);
    pc->nc = atoi(&buf[50]);
    pc->rf = atof(&buf[55]);
    pc->used = 0;
    for (i=0; i<pc->nc/3 + (pc->nc%3)?1:0; i++) {
        rv=fgets(buf, 90, f);
        if (rv==NULL) { return(-1); }
        for (j=0; j<90; j++) { if (buf[j]=='D' || buf[j]=='d') buf[j]='e'; }
        ret=sscanf(buf, "%lf %lf %lf", 
                &(pc->c[3*i]), &(pc->c[3*i+1]), &(pc->c[3*i+2]));
        if (ret!=3) { return(-1); }
    }

    return(0);

}

int read_pc(FILE *f, struct polyco *pc, const char *psr, int mjd, double fmjd) {

    /* Read through until we get to right psr, mjd */
    int done=0, nomatch=0;
    int i, j;
    char *rv;
    int ret;
    char buf[90];
    float tdiff;
    while (!done) {
        /* Read in polyco chunk */
        rv = fgets(buf, 90, f);
        if (rv==NULL) { done=1; nomatch=1; continue; }
        strncpy(pc->psr, &buf[0], 10);  pc->psr[10] = '\0';
        pc->mjd = atoi(&buf[31]);
        pc->fmjd = atof(&buf[39]);
        if ((rv=strchr(pc->psr, ' '))!=NULL) { *rv='\0'; }
        rv = fgets(buf,90,f);
        pc->rphase = fmod(atof(&buf[0]),1.0);
        pc->f0 = atof(&buf[20]);
        pc->nsite = atoi(&buf[42]);
        pc->nmin = atoi(&buf[43]);
        pc->nc = atoi(&buf[50]);
        pc->rf = atof(&buf[55]);
        for (i=0; i<pc->nc/3 + (pc->nc%3)?1:0; i++) {
            rv=fgets(buf, 90, f);
            if (rv==NULL) { return(-1); }
            for (j=0; j<90; j++) { if (buf[j]=='D' || buf[j]=='d') buf[j]='e'; }
            ret=sscanf(buf, "%lf %lf %lf", 
                    &(pc->c[3*i]), &(pc->c[3*i+1]), &(pc->c[3*i+2]));
            if (ret!=3) { return(-1); }
        }
        /* check for correct psr - null psrname matches any */
        if (psr!=NULL) { if (strcmp(pc->psr, psr)!=0) { continue; } }
        tdiff = 1440.0*((double)(mjd-pc->mjd) + (fmjd-pc->fmjd));
        if (fabs(tdiff) > (float)pc->nmin/2.0) { continue; }
        done=1;
    }

    return(-1*nomatch);

}

/* Reads all polycos in a file, mallocs space for them, returns
 * number found
 */
int read_all_pc(FILE *f, struct polyco **pc) {
    int rv, npc=0;
    do { 
        *pc = (struct polyco *)realloc(*pc, sizeof(struct polyco) * (npc+1));
        rv = read_one_pc(f, &((*pc)[npc]));
        npc++;
    } while (rv==0); 
    npc--; // Final "read" is really a error or EOF.
    return(npc);
}

/* Select appropriate polyco set */
int select_pc(const struct polyco *pc, int npc, const char *psr,
        int imjd, double fmjd) {
    int ipc;
    const char *tmp = psr;
    if (psr!=NULL)
        if (tmp[0]=='J' || tmp[0]=='B') tmp++;
    // Verbose
    //fprintf(stderr, "Looking for polycos with src='%s' imjd=%d fmjd=%f\n",
    //        tmp, imjd, fmjd);
    for (ipc=0; ipc<npc; ipc++) {
        //fprintf(stderr, "  read src='%s' imjd=%d fmjd=%f span=%d\n",
        //        pc[ipc].psr, pc[ipc].mjd, pc[ipc].fmjd, pc[ipc].nmin);
        if (psr!=NULL) { if (strcmp(pc[ipc].psr,tmp)!=0) { continue; } }
        if (pc_out_of_range(&pc[ipc],imjd,fmjd)==0) { break; }
    }
    if (ipc<npc) { return(ipc); }
    return(-1);
}

/* Compute pulsar phase given polyco struct and mjd */
double psr_phase(const struct polyco *pc, int mjd, double fmjd, double *freq,
        long long *pulsenum) {
    double dt = 1440.0*((double)(mjd-pc->mjd)+(fmjd-pc->fmjd));
    int i;
    double phase = pc->c[pc->nc-1];
    double f = 0.0;
    if (fabs(dt)>(double)pc->nmin/2.0) { return(-1.0); }
    for (i=pc->nc-1; i>0; i--) {
        phase = dt*(phase) + pc->c[i-1];
        f = dt*(f) + (double)i*pc->c[i];
    }
    f = pc->f0 + (1.0/60.0)*f;
    phase += pc->rphase + dt*60.0*pc->f0;
    if (freq!=NULL) { *freq = f; }
    if (pulsenum!=NULL) { 
        long long n = pc->rphase_int;
        n += (long long)(phase - fmod(phase,1.0));
        phase = fmod(phase,1.0);
        if (phase<0.0) { phase += 1.0; n--; }
        *pulsenum = n; 
    }
    return(phase);
}

double psr_fdot(const struct polyco *pc, int mjd, double fmjd, double *fdot) {
    double dt = 1440.0*((double)(mjd-pc->mjd)+(fmjd-pc->fmjd));
    if (fabs(dt)>(double)pc->nmin/2.0) { return(-1.0); }
    double fd=0.0;
    int i;
    for (i=pc->nc-1; i>1; i--) {
        fd = dt*fd + ((double)i)*((double)i-1.0)*pc->c[i];
    }
    fd /= 60.0;
    if (fdot!=NULL) { *fdot=fd; }
    return(fd);
}

double psr_phase_avg(const struct polyco *pc, int mjd, 
        double fmjd1, double fmjd2) {
    double dt1 = 1440.0*((double)(mjd-pc->mjd)+(fmjd1-pc->fmjd));
    double dt2 = 1440.0*((double)(mjd-pc->mjd)+(fmjd2-pc->fmjd));
    if (fabs(dt1)>(double)pc->nmin/2.0) { return(-1.0); }
    if (fabs(dt2)>(double)pc->nmin/2.0) { return(-1.0); }
    double pavg;
    int i;
    double tmp1=0.0, tmp2=0.0;
    for (i=pc->nc-1; i>=0; i--) {
        tmp1 = dt1*tmp1 + pc->c[i]/((double)i+1.0);
        tmp2 = dt2*tmp2 + pc->c[i]/((double)i+1.0);
    }
    tmp1 *= dt1; tmp2 *= dt2;
    pavg = (tmp2-tmp1)/(dt2-dt1) + pc->rphase + (dt1+dt2)*60.0*pc->f0/2.0;
    return(pavg);
}

int pc_range_check(const struct polyco *pc, int mjd, double fmjd) {
    double dt;
    dt = (double)(mjd - pc->mjd) + (fmjd - pc->fmjd);
    dt *= 1440.0;
    if (dt < -1.0*(double)pc->nmin/2.0) { return(-1); }
    else if (dt > (double)pc->nmin/2.0) { return(1); }
    else { return(0); }
}

int pc_out_of_range(const struct polyco *pc, int mjd, double fmjd) {
    double dt;
    dt = (double)(mjd - pc->mjd) + (fmjd - pc->fmjd);
    dt *= 1440.0;
    if (fabs(dt)>(double)pc->nmin/2.0) { return(1); }
    return(0);
}

int pc_out_of_range_sloppy(const struct polyco *pc, int mjd, double fmjd, 
        double slop) {
    double dt;
    dt = (double)(mjd - pc->mjd) + (fmjd - pc->fmjd);
    dt *= 1440.0;
    if (fabs(dt)>slop*(double)pc->nmin/2.0) { return(1); }
    return(0);
}

/* Check whether or not two polyco structs are the same */
int polycos_differ(const struct polyco *p1, const struct polyco *p2) {
    // Could add more tests as needed
    if (strncmp(p1->psr, p2->psr,15)!=0) return(1);
    if (p1->mjd!=p2->mjd) return(1);
    if (p1->fmjd!=p2->fmjd) return(1);
    if (p1->rf!=p2->rf) return(1);
    if (p1->nsite!=p2->nsite) return(1);
    if (p1->nmin!=p2->nmin) return(1);
    if (p1->nc!=p2->nc) return(1);
    return(0);
}

/* Convert telescope name to tempo code */
char telescope_name_to_code(const char *name) {

    /* Assume a 1-char input is already a code */
    if (strlen(name)==1) { return(name[0]); }

    /* Add to these as needed .. */
    if (strcasecmp(name, "GBT")==0) return('1');

    if (strcasecmp(name, "GB43m")==0) return('a');
    if (strcasecmp(name, "GB 43m")==0) return('a');
    if (strncasecmp(name, "GB140",5)==0) return('a');
    if (strncasecmp(name, "GB 140",6)==0) return('a');

    if (strncasecmp(name, "Arecibo",7)==0) return('3');
    if (strcasecmp(name, "AO")==0) return('3');

    /* Not found, return null */
    return('\0');
}

/* Generate polycos from a parfile */
#define make_polycos_cleanup() do {\
    unlink("pulsar.par");\
    unlink("polyco.dat");\
    unlink("tz.in");\
    chdir(origdir);\
    free(origdir);\
    rmdir(tmpdir);\
} while (0)
int make_polycos(const char *parfile, struct hdrinfo *hdr,
        char *src, struct polyco **pc) {
        
    /* Open parfile */
    FILE *pf = fopen(parfile, "r");
    if (pf==NULL) {
        fprintf(stderr, "make_polycos: Error opening parfile %s\n", parfile);
        return(-1);
    }

    /* Generate temp directory */
    char tmpdir[] = "/tmp/polycoXXXXXX";
    if (mkdtemp(tmpdir)==NULL) {
        fprintf(stderr, "make_polycos: Error generating temp dir.\n");
        fclose(pf);
        return(-1);
    }

    /* change to temp dir */
    char *origdir = getcwd(NULL,0);
    chdir(tmpdir);

    /* Open temp dir */
    char fname[256];
    sprintf(fname, "%s/pulsar.par", tmpdir);
    FILE *fout = fopen(fname, "w");
    if (fout==NULL) {
        fprintf(stderr, "make_polycos: Error writing to temp dir.\n");
        fclose(pf);
        make_polycos_cleanup();
        return(-1);
    }

    /* Get source name, copy file */
    char line[256], parsrc[32]="", *key, *val, *saveptr, *ptr;
    while (fgets(line,256,pf)!=NULL) {
        fprintf(fout, "%s", line);
        while ((ptr=strchr(line,'\t'))!=NULL) *ptr=' ';
        if ((ptr=strrchr(line,'\n')) != NULL) *ptr='\0'; 
        key = strtok_r(line, " ", &saveptr);
        val = strtok_r(NULL, " ", &saveptr);
        if (key==NULL || val==NULL) continue; 
        if (strncmp(key, "PSR", 3)==0) { 
            // J or B is bad here?
            if (val[0]=='J' || val[0]=='B') val++;
            strcpy(parsrc, val); 
        }
    }
    fclose(pf);
    fclose(fout);
    if (parsrc[0]=='\0') {
        fprintf(stderr, "make_polycos: Couldn't find source name in %s\n",
                parfile);
        make_polycos_cleanup();
        return(-1);
    }
    if (src!=NULL) { strcpy(src,parsrc); }

    /* Get telescope character */
    char tcode = telescope_name_to_code(hdr->telescope);
    if (tcode=='\0') {
        fprintf(stderr, "make_polycos: Unrecognized telescope name (%s)\n",
                hdr->telescope);
        make_polycos_cleanup();
        return(-1);
    }

    /* Write tz.in */
    sprintf(fname, "%s/tz.in", tmpdir);
    fout = fopen(fname, "w");
    if (fout==NULL) { 
        fprintf(stderr, "make_polycos: Error opening tz.in for write.\n");
        make_polycos_cleanup();
        return(-1);
    }
    fprintf(fout, "%c 12 30 15 %.5f\n\n\n%s\n",
            tcode, hdr->fctr, parsrc);
    fclose(fout);

    /* Call tempo */
    int mjd0, mjd1;
    mjd0 = (int)hdr->MJD_epoch;
    mjd1 = (int)(hdr->MJD_epoch + hdr->scanlen/86400.0 + 0.5);
    if (mjd1==mjd0) mjd1++;
    sprintf(line, "echo %d %d | tempo -z -f pulsar.par > /dev/null",
            mjd0-1, mjd1);
    system(line);

    /* Read polyco file */
    FILE *pcfile = fopen("polyco.dat", "r");
    if (pcfile==NULL) {
        fprintf(stderr, "make_polycos: Error reading polyco.dat\n");
        make_polycos_cleanup();
        return(-1);
    }
    int npc = read_all_pc(pcfile, pc);
    fclose(pcfile);

    /* Clean up */
    make_polycos_cleanup();

    return(npc);
}

