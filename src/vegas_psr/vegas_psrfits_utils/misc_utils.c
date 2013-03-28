#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <string.h>
#include <ctype.h>
#include <unistd.h>

/* like strcpy, except guaranteed to work with overlapping strings      */
#define strMove(d,s) memmove(d,s,strlen(s)+1)

char *rmtrail(char *str)
/* Removes trailing space from a string */
{
    int i;
    
    if (str && 0 != (i = strlen(str))) {
        while (--i >= 0) {
            if (!isspace(str[i]))
                break;
        }
        str[++i] = '\0';
    }
    return str;
}


char *rmlead(char *str)
/* Removes leading space from a string */
{
    char *obuf;
    
    if (str) {
        for (obuf = str; *obuf && isspace(*obuf); ++obuf);
        if (str != obuf) strMove(str, obuf);
    }
    return str;
}


char *remove_whitespace(char *str)
/* Remove leading and trailing space from a string */
{
    return rmlead(rmtrail(str));
}


char *strlower(char *str)
/* Convert a string to lower case */
{
    char *ss;
    
    if (str) {
        for (ss = str; *ss; ++ss)
            *ss = tolower(*ss);
    }
    return str;
}


void split_path_file(char *input, char **path, char **file)
/* This routine splits an input string into a path and */
/* a filename.  Since it allocates the memory for the  */
/* path and filename dynamically, the calling program  */
/* must free both "path" and "file".                   */
{
    char *sptr = NULL, stmp[200];
    unsigned int len, pathlen = 0, filelen = 0;
    
    len = strlen(input);
    sptr = strrchr(input, '/');
    if (sptr == NULL) {
        getcwd(stmp, 200);
        if (stmp == NULL) {
            printf("\nCurrent directory name is too long.\n");
            printf("Exiting\n\n");
            exit(1);
        }
        pathlen = strlen(stmp);
        *path = (char *) calloc(pathlen + 1, sizeof(char));
        *file = (char *) calloc(len + 1, sizeof(char));
        strcpy(*path, stmp);
        strncpy(*file, input, len);
    } else {
        pathlen = sptr - input;
        filelen = len - pathlen - 1;
        *path = (char *) calloc(pathlen + 1, sizeof(char));
        *file = (char *) calloc(filelen + 1, sizeof(char));
        strncpy(*path, input, pathlen);
        strncpy(*file, sptr + 1, filelen);
    }
}


int split_root_suffix(char *input, char **root, char **suffix)
/* This routine splits an input string into a root name */
/* + suffix.  Since it allocates the memory for the     */
/* root and suffix dynamically, the calling program     */
/* must free both "root" and "suffix".                  */
/* If the routine finds a suffix, it returns 1, else 0. */
{
    char *sptr = NULL;
    unsigned int len, rootlen = 0, suffixlen = 0;
    
    len = strlen(input);
    sptr = strrchr(input, '.');
    if (sptr == NULL) {
        *root = (char *) calloc(len + 1, sizeof(char));
        strncpy(*root, input, len);
        return 0;
    } else {
        rootlen = sptr - input;
        *root = (char *) calloc(rootlen + 1, sizeof(char));
        strncpy(*root, input, rootlen);
        suffixlen = len - rootlen - 1;
        *suffix = (char *) calloc(suffixlen + 1, sizeof(char));
        strncpy(*suffix, sptr + 1, suffixlen);
        return 1;
    }
}


void strtofilename(char *string)
/* Trim spaces off the end of *input and convert */
/* all other spaces into underscores.            */
{
    int ii;
    
    ii = strlen(string) - 1;
    do {
        if (string[ii] == ' ')
            string[ii] = '\0';
        else
            break;
    } while (ii--);
    do {
        if (string[ii] == ' ')
            string[ii] = '_';
    } while (ii--);
}


double delay_from_dm(double dm, double freq_emitted)
/* Return the delay in seconds caused by dispersion, given  */
/* a Dispersion Measure (dm) in cm-3 pc, and the emitted    */
/* frequency (freq_emitted) of the pulsar in MHz.           */
{
    return dm / (0.000241 * freq_emitted * freq_emitted);
}


long long next2_to_n(long long x)
/* Return the first value of 2^n >= x */
{
    long long i = 1;
    
    while (i < x)
        i <<= 1;
    return i;
}


void avg_std(char *x, int n, double *mean, double *std, int stride)
/* For an unsigned char vector, *x, of length n*stride, this    */
/* routine returns the mean and variance of the n values        */  
/* separated in memory by stride bytes (contiguous is stride=1) */
{
    int ii;
    double an = 0.0, an1 = 0.0, dx, var;

    /*  Modified (29 June 98) C version of the following:        */
    /*  ALGORITHM AS 52  APPL. STATIST. (1972) VOL.21, P.226     */
    /*  Returned values were checked with Mathematica 3.01       */
    
    if (n < 1) {
        printf("\vVector length must be > 0 in avg_var().  Exiting\n");
        exit(1);
    } else {
        *mean = (double) x[0];
        var = 0.0;
    }
    
    for (ii = 1; ii < n; ii++) {
        an = (double) (ii + 1);
        an1 = (double) (ii);
        dx = ((double) x[ii*stride] - *mean) / an;
        var += an * an1 * dx * dx;
        *mean += dx;
    }
    
    if (n > 1) {
        var /= an1;
        *std = sqrt(var);
    } else {
        *std = 0.0;
    }
    
    return;
}


static int TOMS_gcd(int a, int b)
/* Return the greatest common denominator of 'a' and 'b' */
{
    int r;
    do {
        r = a % b;
        a = b;
        b = r;
    } while (r != 0);
    
    return a;
}


short transpose_bytes(unsigned char *a, int nx, int ny, unsigned char *move,
                      int move_size)
/*
 * TOMS Transpose.  Revised version of algorithm 380.
 *
 * These routines do in-place transposes of arrays.
 *
 * [ Cate, E.G. and Twigg, D.W., ACM Transactions on Mathematical Software,
 *   vol. 3, no. 1, 104-110 (1977) ]
 *
 * C version by Steven G. Johnson. February 1997.
 *
 * "a" is a 1D array of length ny*nx which contains the nx x ny matrix to be
 * transposed.  "a" is stored in C order (last index varies fastest).  move
 * is a 1D array of length move_size used to store information to speed up
 * the process.  The value move_size=(ny+nx)/2 is recommended.
 *
 * The return value indicates the success or failure of the routine. Returns 0
 * if okay, -1 if ny or nx < 0, and -2 if move_size < 1. The return value
 * should never be positive, but it it is, it is set to the final position in
 * a when the search is completed but some elements have not been moved.
 *
 * Note: move[i] will stay zero for fixed points.
 */
{
   int i, j, im, mn;
   unsigned char b, c, d;
   int ncount;
   int k;

   /* check arguments and initialize: */
   if (ny < 0 || nx < 0)
      return -1;
   if (ny < 2 || nx < 2)
      return 0;
   if (move_size < 1)
      return -2;

   if (ny == nx) {
      /*
       * if matrix is square, exchange elements a(i,j) and a(j,i):
       */
      for (i = 0; i < nx; ++i)
         for (j = i + 1; j < nx; ++j) {
            b = a[i + j * nx];
            a[i + j * nx] = a[j + i * nx];
            a[j + i * nx] = b;
         }
      return 0;
   }
   ncount = 2;                  /* always at least 2 fixed points */
   k = (mn = ny * nx) - 1;

   for (i = 0; i < move_size; ++i)
      move[i] = 0;

   if (ny >= 3 && nx >= 3)
      ncount += TOMS_gcd(ny - 1, nx - 1) - 1;   /* # fixed points */

   i = 1;
   im = ny;

   while (1) {
      int i1, i2, i1c, i2c;
      int kmi;

    /** Rearrange the elements of a loop
        and its companion loop: **/

      i1 = i;
      kmi = k - i;
      b = a[i1];
      i1c = kmi;
      c = a[i1c];

      while (1) {
         i2 = ny * i1 - k * (i1 / nx);
         i2c = k - i2;
         if (i1 < move_size)
            move[i1] = 1;
         if (i1c < move_size)
            move[i1c] = 1;
         ncount += 2;
         if (i2 == i)
            break;
         if (i2 == kmi) {
            d = b;
            b = c;
            c = d;
            break;
         }
         a[i1] = a[i2];
         a[i1c] = a[i2c];
         i1 = i2;
         i1c = i2c;
      }
      a[i1] = b;
      a[i1c] = c;

      if (ncount >= mn)
         break;                 /* we've moved all elements */

    /** Search for loops to rearrange: **/

      while (1) {
         int max;

         max = k - i;
         ++i;
         if (i > max)
            return i;
         im += ny;
         if (im > k)
            im -= k;
         i2 = im;
         if (i == i2)
            continue;
         if (i >= move_size) {
            while (i2 > i && i2 < max) {
               i1 = i2;
               i2 = ny * i1 - k * (i1 / nx);
            }
            if (i2 == i)
               break;
         } else if (!move[i])
            break;
      }
   }

   return 0;
}
