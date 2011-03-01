from guppi_utils import *
from astro_utils import current_MJD
from optparse import OptionParser

# Cmd line
par = OptionParser()
par.add_option("-c", "--cal", dest="cal", 
    help="Set parameters for a cal scan", action="store_true", 
    default=False)
(opt,arg) = par.parse_args()

# Connect to status buf
g = guppi_status()

# Fill values from gbtstatus database
g.update_with_gbtstatus()

g.update("SCANNUM", 1)
g.update("OBS_MODE", "SEARCH")

# Cal-specific params
g.update("CAL_FREQ", 25.0)
g.update("CAL_DCYC", 0.5)
g.update("CAL_PHS", 0.0)
if (opt.cal):
    g.update("SCANLEN", 120.0)
    g.update("BASENAME", "guppi_test_%s_%04d_cal"%(g['SRC_NAME'], g['SCANNUM']))
    g.update("CAL_MODE", "ON")
else:
    g.update("SCANLEN", 1800.0)
    g.update("BASENAME", "guppi_test_%s_%04d"%(g['SRC_NAME'], g['SCANNUM']))
    g.update("CAL_MODE", "OFF")

# GUPPI mode settings
g.update("BACKEND", "GUPPI")
g.update("PKTFMT", "GUPPI")
g.update("OBSBW", 800.0)
g.update("OBSNCHAN", 2048)
g.update("NPOL", 4)
g.update("POL_TYPE", "IQUV")
g.update("NBITS", 8)
g.update("PFB_OVER", 4)
g.update("NBITSADC", 8)
g.update("ACC_LEN", 16)
g.update("TBIN", abs(g['ACC_LEN']*g['OBSNCHAN']/g['OBSBW']*1e-6))
g.update("CHAN_BW", g['OBSBW']/g['OBSNCHAN'])

g.update("OFFSET0", 0.0)
g.update("SCALE0", 1.0)
g.update("OFFSET1", 0.0)
g.update("SCALE1", 1.0)
g.update("OFFSET2", 0.0)
g.update("SCALE2", 1.0)
g.update("OFFSET3", 0.0)
g.update("SCALE3", 1.0)

# Fill in current time, will be replaced with accurate start time
# when datataking starts.
MJD = current_MJD()
MJDd = int(MJD)
MJDf = MJD - MJDd
MJDs = int(MJDf * 86400 + 1e-6)
offs = (MJD - MJDd - MJDs/86400.0) * 86400.0
g.update("STT_IMJD", MJDd)
g.update("STT_SMJD", MJDs)
if offs < 2e-6: offs = 0.0
g.update("STT_OFFS", offs)

g.update_azza()
g.write()
