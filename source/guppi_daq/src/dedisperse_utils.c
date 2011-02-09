/* dedisperse_utils.c
 *
 * Misc helpful functions for coherent dedisp.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "dedisperse_utils.h"

/* Pull a DM value out of a parfile.
 * Negative return value indicates a problem.. 
 */
double dm_from_parfile(const char *parfile) {
    FILE *f = fopen(parfile,"r");
    if (f==NULL) { return(-1.0); }
    char line[256];
    double dm=-2.0;
    char *ptr, *key, *val, *saveptr;
    while (fgets(line, 256, f)!=NULL) {

        // Convert tabs to spaces
        while ((ptr=strchr(line,'\t'))!=NULL) { *ptr=' '; }

        // strip leading whitespace
        ptr = line;
        while (*ptr==' ') { ptr++; }

        // Identify comments or blank lines
        if (line[0]=='\n' || line[0]=='#' || 
                (line[0]=='C' && line[1]==' '))
            continue;

        // Split into key/val (ignore fit flag and error)
        key = strtok_r(line,  " ", &saveptr);
        val = strtok_r(NULL, " ", &saveptr);
        if (key==NULL || val==NULL) continue; // TODO : complain?

        if (strncmp(key, "DM", 3)==0) {
            dm = atof(val);
            break;
        }
    }
    fclose(f);
    return(dm);
}
