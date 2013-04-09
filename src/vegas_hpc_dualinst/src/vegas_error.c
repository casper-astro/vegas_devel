/* vegas_error.c
 *
 * Error handling routine
 */
#include <stdio.h>
#include "vegas_error.h"

/* For now just put it all to stderr.
 * Maybe do something clever like a stack in the future?
 */
void vegas_error(const char *name, const char *msg) {
    fprintf(stderr, "Error (%s): %s\n", name, msg);
    fflush(stderr);
}

void vegas_warn(const char *name, const char *msg) {
    fprintf(stderr, "Warning (%s): %s\n", name, msg);
    fflush(stderr);
}
