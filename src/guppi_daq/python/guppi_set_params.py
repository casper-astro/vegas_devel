import os, sys
from guppi_utils import *
from astro_utils import current_MJD
from optparse import OptionParser

# Parse command line

# Check that something was given on the command line
nargs = len(sys.argv) - 1

# Dict containing list of key/val pairs to update in
# the status shared mem
update_list = {}
def add_param(option, opt_str, val, parser, *args):
    update_list[args[0]] = val

# Func to add a key/val setting option to the command line.
# longopt, short are the command line flags
# name is the shared mem key (ie, SCANNUM, RA_STR, etc)
# type is the value type (string, float, etc)
# help is the help string for -h
par = OptionParser()
def add_param_option(longopt, name, help, type="string", short=None):
    if short!=None:
        par.add_option(short, longopt, help=help, 
                action="callback", callback=add_param, 
                type=type, callback_args=(name,))
    else:
        par.add_option(longopt, help=help, 
                action="callback", callback=add_param, 
                type=type, callback_args=(name,))

# Non-parameter options
par.add_option("-U", "--update", dest="update",
        help="Run in update mode",
        action="store_true", default=False)
par.add_option("-D", "--default", dest="default",
        help="Use all default values",
        action="store_true", default=True)
par.add_option("-f", "--force", dest="force",
        help="Force guppi_set_params to run even if unsafe",
        action="store_true", default=False)
par.add_option("-c", "--cal", dest="cal", 
               help="Setup for cal scan (folding mode)",
               action="store_true", default=False)
par.add_option("-i", "--increment_scan", dest="inc",
               help="Increment scan num",
               action="store_true", default=False)
par.add_option("-I", "--onlyI", dest="onlyI",
               help="Only record total intensity",
               action="store_true", default=False)
par.add_option("--dm", dest="dm", 
        help="Optimize overlap params using given DM",
        action="store", type="float", default=None)

# Parameter-setting options
add_param_option("--scannum", short="-n", 
        name="SCANNUM", type="int",
        help="Set scan number")
add_param_option("--tscan", short="-T",
        name="SCANLEN", type="float",
        help="Scan length (sec)")
add_param_option("--parfile", short="-P",
        name="PARFILE", type="string", 
        help="Use this parfile for folding")
add_param_option("--tfold", short="-t",
        name="TFOLD", type="float",
        help="Fold dump time (sec)")
add_param_option("--bins", short="-b",
        name="NBIN", type="int",
        help="Number of profile bins for folding")
add_param_option("--cal_freq",
        name="CAL_FREQ", type="float",
        help="Frequency of pulsed noise cal (Hz, default 25.0)")
add_param_option("--dstime", 
        name="DS_TIME", type="int",
        help="Downsample in time (int, power of 2)")
add_param_option("--dsfreq", 
        name="DS_FREQ", type="int",
        help="Downsample in freq (int, power of 2)")
add_param_option("--obs", 
        name="OBSERVER", type="string",
        help="Set observers name")
add_param_option("--src", 
        name="SRC_NAME", type="string",
        help="Set observed source name")
add_param_option("--ra", 
        name="RA_STR", type="string",
        help="Set source R.A. (hh:mm:ss.s)")
add_param_option("--dec", 
        name="DEC_STR", type="string",
        help="Set source Dec (+/-dd:mm:ss.s)")
add_param_option("--freq", 
        name="OBSFREQ", type="float",
        help="Set center freq (MHz)")
add_param_option("--bw", 
        name="OBSBW", type="float",
        help="Hardware total bandwidth (MHz)")
add_param_option("--obsnchan", 
        name="OBSNCHAN", type="int",
        help="Number of hardware channels")
add_param_option("--npol", 
        name="NPOL", type="int",
        help="Number of hardware polarizations")
add_param_option("--nchan", 
        name="NCHAN", type="int",
        help="Number of spectral channels per sub-band")
add_param_option("--chan_bw", 
        name="CHAN_BW", type="float",
        help="Width of each spectral bin (resolution)")
add_param_option("--feed_pol", 
        name="FD_POLN", type="string",
        help="Feed polarization type (LIN/CIRC)")
add_param_option("--acc_len", 
        name="ACC_LEN", type="int",
        help="Hardware accumulation length")
add_param_option("--packets", 
        name="PKTFMT", type="string",
        help="UDP packet format")
add_param_option("--host",
        name="DATAHOST", type="string",
        help="IP or hostname of data source")
add_param_option("--datadir", 
        name="DATADIR", type="string",
        help="Data output directory (default: current dir)")
add_param_option("--tsys", 
        name="TSYS", type="float",
        help="System temperature")
add_param_option("--nsubband", 
        name="NSUBBAND", type="int",
        help="Number of sub-bands (1-8)")
add_param_option("--exposure", 
        name="EXPOSURE", type="float",
        help="Required integration time (in seconds)")
add_param_option("--efsampfr", 
        name="EFSAMPFR", type="float",
        help="Effective sampling frequency (after decimation), in Hz")
add_param_option("--fpgaclk", 
        name="FPGACLK", type="float",
        help="FPGA clock rate (in Hz)")
add_param_option("--hwexposr", 
        name="HWEXPOSR", type="float",
        help="Duration of fixed integration on FPGA/GPU [s]")
add_param_option("--filtnep", 
        name="FILTNEP", type="float",
        help="PFB filter noise-equivalent parameters")
add_param_option("--projid", 
        name="PROJID", type="string",
        help="Project ID string")
add_param_option("--elev", 
        name="ELEV", type="float",
        help="Current antenna elevation, above horizon")
add_param_option("--object", 
        name="OBJECT", type="string",
        help="The object currently being viewed")
add_param_option("--bmaj", 
        name="BMAJ", type="float",
        help="Beam major axis length")
add_param_option("--bmin", 
        name="BMIN", type="float",
        help="Beam minor axis length")
add_param_option("--bpa", 
        name="BPA", type="float",
        help="Beam position angle")
add_param_option("--sub0freq", 
        name="SUB0FREQ", type="float",
        help="Centre frequency of sub-band 0")
add_param_option("--sub1freq", 
        name="SUB1FREQ", type="float",
        help="Centre frequency of sub-band 1")
add_param_option("--sub2freq", 
        name="SUB2FREQ", type="float",
        help="Centre frequency of sub-band 2")
add_param_option("--sub3freq", 
        name="SUB3FREQ", type="float",
        help="Centre frequency of sub-band 3")
add_param_option("--sub4freq", 
        name="SUB4FREQ", type="float",
        help="Centre frequency of sub-band 4")
add_param_option("--sub5freq", 
        name="SUB5FREQ", type="float",
        help="Centre frequency of sub-band 5")
add_param_option("--sub6freq", 
        name="SUB6FREQ", type="float",
        help="Centre frequency of sub-band 6")
add_param_option("--sub7freq", 
        name="SUB7FREQ", type="float",
        help="Centre frequency of sub-band 7")
add_param_option("--filenum", 
        name="FILENUM", type="int",
        help="Current file number, in a multifile scan")
add_param_option("--port", 
        name="DATAPORT", type="int",
        help="Port on which to receive packets")


# non-parameter options
par.add_option("--nogbt", dest="gbt", 
               help="Don't pull values from gbtstatus",
               action="store_false", default=True)
par.add_option("--gb43m", dest="gb43m", 
               help="Set params for 43m observing",
               action="store_true", default=False)
par.add_option("--fake", dest="fake",
               help="Set params for fake psr",
               action="store_true", default=False)

(opt,arg) = par.parse_args()

# If extra command line stuff, exit
if (len(arg)>0):
    par.print_help()
    print
    print "guppi_set_params: Unrecognized command line values", arg
    print
    sys.exit(0)

# If nothing was given on the command line, print help and exit
if (nargs==0):
    par.print_help()
    print 
    print "guppi_set_params: No command line options were given, exiting."
    print "  Either specifiy some options, or to use all default parameter" 
    print "  values, run with the -D flag."
    print
    sys.exit(0)

# Check for ongoing observations
if (os.popen("pgrep guppi_daq").read() != ""):
    if (opt.force):
        print "Warning: Proceeding to set params even though datataking is currently running!"
    else:
        print """
guppi_set_params: A GUPPI datataking process appears to be running, exiting.
  If you really want to change the parameters, run again with the --force 
  option.  Note that this will likely cause problems with the current 
  observation.
        """
        sys.exit(1)

# 43m implies nogbt
if (opt.gb43m):
    opt.gbt = False

# Fake psr implies nogbt
if (opt.fake):
    opt.gbt = False

# Attach to status shared mem
g = guppi_status()

# read what's currently in there
g.read()

# If we're not in update mode
if (opt.update == False):

    # These will later get overwritten with gbtstatus and/or
    # command line values
    g.update("SRC_NAME", "unknown")
    g.update("OBSERVER", "unknown")
    g.update("RA_STR", "00:00:00.0")
    g.update("DEC_STR", "+00:00:00.0")
    g.update("TELESCOP", "GBT")
    g.update("FRONTEND", "none")
    g.update("PROJID", "test")
    g.update("FD_POLN", "LIN")
    g.update("TRK_MODE", "TRACK")
    g.update("OBSFREQ", 1200.0)
    g.update("OBSBW", 800.0)

    g.update("OBS_MODE", "SEARCH")
    g.update("CAL_MODE", "OFF")

    g.update("SCANLEN", 8 * 3600.)
    g.update("BACKEND", "GUPPI")
    g.update("PKTFMT", "1SFA")
    g.update("DATAHOST", "bee2-10")
    g.update("DATAPORT", 50000)
    g.update("POL_TYPE", "IQUV")

    g.update("CAL_FREQ", 25.0)
    g.update("CAL_DCYC", 0.5)
    g.update("CAL_PHS", 0.0)

    g.update("OBSNCHAN", 2048)
    g.update("NCHAN", 1024)
    g.update("CHAN_BW", 1000)
    g.update("NPOL", 4)
    g.update("NBITS", 8)
    g.update("PFB_OVER", 4)
    g.update("NBITSADC", 8)
    g.update("ACC_LEN", 16)
    g.update("NRCVR", 2)

    g.update("ONLY_I", 0)
    g.update("DS_TIME", 1)
    g.update("DS_FREQ", 1)

    g.update("TFOLD", 30.0)
    g.update("NBIN", 256)
    g.update("PARFILE", "")

    g.update("OFFSET0", 0.0)
    g.update("SCALE0", 1.0)
    g.update("OFFSET1", 0.0)
    g.update("SCALE1", 1.0)
    #g.update("OFFSET2", 0.5)
    g.update("OFFSET2", 0.0)
    g.update("SCALE2", 1.0)
    #g.update("OFFSET3", 0.5)
    g.update("OFFSET3", 0.0)
    g.update("SCALE3", 1.0)

    g.update("DATADIR", ".")

    #Default settings for VEGAS specific variables
    g.update("TSYS", 0)
    g.update("NSUBBAND", 1)
    g.update("ELEV", 0)
    g.update("OBJECT", "unknown_obj")
    g.update("BMAJ", 0)
    g.update("BMIN", 0)
    g.update("BPA", 0)
    g.update("EXPOSURE", 1e0)
    g.update("PFBRATE", 10e6)
    g.update("SUB0FREQ", 1e9)
    g.update("SUB1FREQ", 1e9)
    g.update("SUB2FREQ", 1e9)
    g.update("SUB3FREQ", 1e9)
    g.update("SUB4FREQ", 1e9)
    g.update("SUB5FREQ", 1e9)
    g.update("SUB6FREQ", 1e9)
    g.update("SUB7FREQ", 1e9)
    g.update("FILENUM", 0)

    # Pull from gbtstatus if needed
    if (opt.gbt):
        g.update_with_gbtstatus()

    # Current time (updated for VEGAS)
    MJD = current_MJD()
    g.update("STTMJD", MJD)

    # Misc
    g.update("LST", 0)


# Any 43m-specific settings
if (opt.gb43m):
    g.update("TELESCOP", "GB43m")
    g.update("FRONTEND", "43m_rcvr")
    g.update("FD_POLN", "LIN")

# Any fake psr-specific settings
if (opt.fake):
    g.update("SRC_NAME", "Fake_PSR")
    g.update("FRONTEND", "none")
    g.update("OBSFREQ", 1000.0)
    g.update("FD_POLN", "LIN")

# Total intensity mode
if (opt.onlyI):
    g.update("ONLY_I", 1)

# Cal mode
if (opt.cal):
    g.update("OBS_MODE", "CAL")
    g.update("CAL_MODE", "ON")
    g.update("TFOLD", 10.0)
    g.update("SCANLEN", 120.0)

# Scan number
try:
    scan = g["SCANNUM"]
    if (opt.inc):
        g.update("SCANNUM", scan+1)
except KeyError:
    g.update("SCANNUM", 1)

# Observer name
try:
    obsname = g["OBSERVER"]
except KeyError:
    obsname = "unknown"

if (obsname=="unknown"):
    try:
        username = os.environ["LOGNAME"]
    except KeyError:
        username = os.getlogin()
    g.update("OBSERVER", username)

# Apply explicit command line values
# These will always take precedence over defaults now
for (k,v) in update_list.items():
    g.update(k,v)

# Apply to shared mem
g.write()

# Update any derived parameters:

# Base file name
if (opt.cal):
    base = "guppi_%5d_%s_%04d_cal" % (g['STTMJD'], 
            g['SRC_NAME'], g['SCANNUM'])
else:
    base = "vegas_%5d_%s_%04d" % (g['STTMJD'], 
            g['SRC_NAME'], g['SCANNUM'])
g.update("BASENAME", base)

# Time res, channel bw
g.update("TBIN", abs(g['ACC_LEN']*g['OBSNCHAN']/g['OBSBW']*1e-6))
# g.update("CHAN_BW", g['OBSBW']/g['OBSNCHAN'])

# Az/el
g.update_azza()

# Overlap params for coherent dedisp
if (opt.dm!=None):
    lofreq_ghz = (g['OBSFREQ']-abs(g['OBSBW']/2.0))/1.0e3
    overlap_samp = 8.3 * g['CHAN_BW']**2 / lofreq_ghz**3 * opt.dm
    #round_fac = 128 # XXX this depends on nchan...
    round_fac = 8192 # XXX this depends on nchan...
    overlap_r = round_fac * (int(overlap_samp)/round_fac + 1)
    # Rough optimization for fftlen
    fftlen = 16*1024
    if overlap_r<=1024: fftlen=32*1024
    elif overlap_r<=2048: fftlen=64*1024
    elif overlap_r<=16*1024: fftlen=128*1024
    elif overlap_r<=64*1024: fftlen=256*1024
    while fftlen<2*overlap_r: fftlen *= 2
    # or, hard-code 128k ffts:
    #fftlen = 128*1024
    databuf_mb = 128
    #databuf_mb = 64
    #databuf_mb = 32
    npts_max_per_chan = databuf_mb*1024*1024/4/g['OBSNCHAN']
    nfft = (npts_max_per_chan - overlap_r)/(fftlen - overlap_r)
    npts = nfft*(fftlen-overlap_r) + overlap_r
    blocsize = npts*g['OBSNCHAN']*4
    g.update("FFTLEN", fftlen)
    g.update("OVERLAP", overlap_r)
    g.update("BLOCSIZE", blocsize)
    g.update("CHAN_DM", opt.dm);

# Apply back to shared mem
g.write()
