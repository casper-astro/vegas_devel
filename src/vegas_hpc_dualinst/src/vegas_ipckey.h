/*
 *  vegas_ipckey.h
 *
 *  Declare functions used to get IPC keys for VEGAS.
 */

#ifndef _VEGAS_IPCKEY_H
#define _VEGAS_IPCKEY_H

#include <sys/ipc.h>

#define VEGAS_KEY_ERROR ((key_t)-1)
/*
 * Get the base key to use for vegas databufs.
 *
 * If VEGAS_DATABUF_KEY is defined in the environment, its value is used as the
 * base databuf key.  Otherwise, the base key is obtained by calling the ftok
 * function, using the value of $VEGAS_KEYFILE, if defined, or $HOME from the
 * environment or, if $HOME is not defined, "/tmp" for the pathname and a
 * databuf specific proj_id derived from the lower 6 bits of the instance_id.
 * Use of instance_id allows a user to run multiple instances of a pipeline
 * without having to alter the environment (even from within the same process).
 *
 * By default (i.e. VEGAS_DATABUF_KEY and VEGAS_KEYFILE do not exist in the
 * environment), this will create and connect to a user specific set of shared
 * memory buffers (provided $HOME exists in the environment), but if desired
 * users can connect to any other set of memory buffers by setting
 * VEGAS_KEYFILE appropraitely.
 *
 * VEGAS_KEY_ERROR is returned on error.
 */
key_t vegas_databuf_key(int instance_id);

/*
 * Get the base key to use for the vegas status buffer.
 *
 * If VEGAS_STATUS_KEY is defined in the environment, its value is used as
 * the base databuf key.  Otherwise, the base key is obtained by calling the
 * ftok function, using the value of $VEGAS_KEYFILE, if defined, or $HOME from
 * the environment or, if $HOME is not defined, "/tmp" for the pathname and a
 * status buffer specific proj_id derived from the lower 6 bits of the
 * instance_id.  Use of instance_id allows a user to run multiple instances of
 * a pipeline without having to alter the environment (even from within the
 * same process).
 *
 * By default (i.e. VEGAS_STATUS_KEY and VEGAS_KEYFILE do not exist in the
 * environment), this will create and connect to a user specific set of shared
 * memory buffers (provided $HOME exists in the environment), but if desired
 * users can connect to any other set of memory buffers by setting
 * VEGAS_KEYFILE appropraitely.
 *
 * VEGAS_KEY_ERROR is returned on error.
 */
key_t vegas_status_key(int instance_id);

#endif // _VEGAS_IPCKEY_H
