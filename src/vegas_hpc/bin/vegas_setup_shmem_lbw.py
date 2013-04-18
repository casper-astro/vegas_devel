#!/opt/vegas/bin/python

# vegas_setup_shmem_lbw.py

import os
import sys
import getopt
import commands

# function definitions
def PrintUsage(ProgName):
    "Prints usage information."
    print "Usage: " + ProgName + " [options]"
    print "    -h  --help                 Display this usage information"
    print "    -a  --adcclk <value>       ADC clock rate in MHz"
    print "    -s  --nsubband <value>     Number of sub-bands in data"
    print "    -c  --nchan <value>        Number of channels"
    print "    -g  --gpuint <value>       Integration time on GPU in ms"
    print "    -t  --totint <value>       Total integration time in ms"
    return

# constants
# ASSUMPTION: using ASIAA ADC that is dual-data-rate, decimation factor fixed
DecFactor = 128     # decimation factor for both pols.

# other config info
ROACH_10GbE_IP = "10.0.0.4"
Port = "60000"

# get the command line arguments
ProgName = sys.argv[0]
OptsShort = "ha:s:c:g:t:"
OptsLong = ["help", "adcclk=", "nsubband=", "nchan=", "gpuint=", "totint="]

# check if the minimum expected number of arguments has been passed
# to the program
if (1 == len(sys.argv)):
    sys.stderr.write("ERROR: No arguments passed to the program!\n")
    PrintUsage(ProgName)
    sys.exit(1)

# get the arguments using the getopt module
try:
    (Opts, Args) = getopt.getopt(sys.argv[1:], OptsShort, OptsLong)
except getopt.GetoptError, ErrMsg:
    # print usage information and exit
    sys.stderr.write("ERROR: " + str(ErrMsg) + "!\n")
    PrintUsage(ProgName)
    sys.exit(1)

# parse the arguments
for o, a in Opts:
    if o in ("-h", "--help"):
        PrintUsage(ProgName)
        sys.exit()
    elif o in ("-a", "--adcclk"):
        adcClock = float(a)
    elif o in ("-s", "--nsubband"):
        nSubBand = int(a)
    elif o in ("-c", "--nchan"):
        nChan = int(a)
    elif o in ("-g", "--gpuint"):
        gpuInt = float(a)
    elif o in ("-t", "--totint"):
        totInt = float(a)
    else:
        PrintUsage(ProgName)
        sys.exit(1)

if totInt < gpuInt:
    sys.stderr.write("ERROR: Total integration time must be greater than or equal to GPU integration time!\n")
    sys.exit(1)

dirVEGAS = os.getenv("VEGAS_DIR")

(Status, Output) = commands.getstatusoutput(dirVEGAS + "/bin/vegas_init_shmem")

(Status, Output) = commands.getstatusoutput("python2.7"                                                      \
                                            + " " + dirVEGAS + "/python/vegas_set_params.py"                 \
                                            + " -D --nogbt")
if (0 != Status):
    sys.stderr.write("ERROR: vegas_set_params.py failed. " + Output + "\n")
    sys.exit(1)

# calculate parameters from user input
fpgaClock = (adcClock * 1e6) / 8
subBandBW = (2 * adcClock * 1e6) / DecFactor
chanBW = subBandBW / nChan

(Status, Output) = commands.getstatusoutput("python2.7"                                                      \
                                            + " " + dirVEGAS + "/python/vegas_set_params.py"                 \
                                            + " -U --nogbt"                                                  \
                                            + " --host=" + ROACH_10GbE_IP                                    \
                                            + " --port=" + Port                                              \
                                            + " --packets=SPEAD"                                             \
                                            + " --npol=2"                                                    \
                                            + " --nchan=" + str(nChan)                                       \
                                            + " --chan_bw=" + str(chanBW)                                    \
                                            + " --nsubband=" + str(nSubBand)                                 \
                                            + " --sub0freq=2.4e9"                                            \
                                            + " --sub1freq=2.5e9"                                            \
                                            + " --sub2freq=2.6e9"                                            \
                                            + " --sub3freq=2.7e9"                                            \
                                            + " --sub4freq=2.8e9"                                            \
                                            + " --sub5freq=2.9e9"                                            \
                                            + " --sub6freq=3.0e9"                                            \
                                            + " --sub7freq=3.1e9"                                            \
                                            + " --exposure=" + str(totInt)                                   \
                                            + " --fpgaclk=" + str(fpgaClock)                                 \
                                            + " --efsampfr=46.875e6"                                         \
                                            + " --hwexposr=" + str(gpuInt)                                   \
                                            + " --obsmode=LBW")
if (0 != Status):
    sys.stderr.write("ERROR: vegas_set_params.py failed. " + Output + "\n")
    sys.exit(1)

