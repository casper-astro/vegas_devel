#!/usr/bin/env python
import time
from guppi_daq.guppi_utils import *

# Attach to status shared mem
g = guppi_status()

while (1):
    try:
        g.read()
        g.update_with_gbtstatus()
        g.write()
    except:
        pass
    time.sleep(1)
