from guppi_utils import *
from astro_utils import current_MJD
from optparse import OptionParser

# Parse options
par = OptionParser()
par.add_option("-p", "--npol", type="int", dest="npol", 
        help="Number of polarizations", default=2)
(opt,arg) = par.parse_args()
npol = opt.npol

g = guppi_status()

g.update("SCANNUM", 9)

g.update("SRC_NAME", "B0329+54")
g.update("RA_STR", "03:32:59.36")
g.update("DEC_STR", "+54:34:43.6")
#g.update("SRC_NAME", "B1937+21")
#g.update("RA_STR", "19:39:38.560")
#g.update("DEC_STR", "21:34:59.143")
#g.update("SRC_NAME", "B0950+08")
#g.update("RA_STR", "09:53:09.309")
#g.update("DEC_STR", "+07:55:35.75")
#g.update("SRC_NAME", "J1012+5307")
#g.update("RA_STR", "10:12:33.436")
#g.update("DEC_STR", "+53:07:02.40")

g.update("OBS_MODE", "SEARCH")
g.update("SCANLEN", 3600.0)

g.update("TELESCOP", "GB43m")
g.update("OBSERVER", "GUPPI Crew")
g.update("FRONTEND", "None")
g.update("PROJID", "first light tests")
g.update("FD_POLN", "LIN")

g.update("OBSFREQ", 1420.0)

g.update("NRCVR", 2)

g.update("TRK_MODE", "TRACK")
g.update("CAL_MODE", "OFF")
g.update("OFFSET0", 0.0)
g.update("SCALE0", 1.0)
g.update("OFFSET1", 0.0)
g.update("SCALE1", 1.0)
g.update("OFFSET2", 0.0)
g.update("SCALE2", 1.0)
g.update("OFFSET3", 0.0)
g.update("SCALE3", 1.0)

# parkes spectrometer values
g.update("BACKEND", "ParSpec");
g.update("PKTFMT", "PARKES");
g.update("NBITS", 8)
g.update("PFB_OVER", 2)
g.update("NBITSADC", 8)
g.update("OBSBW", -400.0)

if npol==2:
    g.update("NPOL", 2);
    g.update("OBSNCHAN", 1024);
    g.update("POL_TYPE", "AABB");
    g.update("ACC_LEN", 13)
if npol==4:
    g.update("NPOL", 4);
    g.update("OBSNCHAN", 512);
    g.update("POL_TYPE", "IQUV");
    g.update("ACC_LEN", 13)

g.update("BASENAME", "parspec_test_%s_%04d"%(g['SRC_NAME'], g['SCANNUM']))

g.update("TBIN", abs(g['ACC_LEN']*g['OBSNCHAN']/g['OBSBW']*1e-6))
g.update("CHAN_BW", g['OBSBW']/g['OBSNCHAN'])

if (1):  # in case we don't get a real start time
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
