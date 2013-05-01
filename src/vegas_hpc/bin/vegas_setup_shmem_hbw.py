#!/opt/vegas/bin/python

# vegas_setup_shmem_hbw.py

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
    print "    -c  --nchan <value>        Number of channels"
    print "    -t  --totint <value>       Total integration time in ms"
    return

# other config info
ROACH_10GbE_IP = "10.0.0.4"
Port = "60000"

# get the command line arguments
ProgName = sys.argv[0]
OptsShort = "ha:c:t:"
OptsLong = ["help", "adcclk=", "nchan=", "totint="]

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
    elif o in ("-c", "--nchan"):
        nChan = int(a)
    elif o in ("-t", "--totint"):
        totInt = float(a)
        totInt = totInt / 1000
    else:
        PrintUsage(ProgName)
        sys.exit(1)

dirVEGAS = os.getenv("VEGAS_DIR")

(Status, Output) = commands.getstatusoutput(dirVEGAS + "/bin/vegas_init_shmem")

(Status, Output) = commands.getstatusoutput("python2.5"                                                      \
                                            + " " + dirVEGAS + "/python/vegas_set_params.py"                 \
                                            + " -D --nogbt")
if (0 != Status):
    sys.stderr.write("ERROR: vegas_set_params.py failed. " + Output + "\n")
    sys.exit(1)

# calculate parameters from user input
fpgaClock = (adcClock * 1e6) / 8
BW = adcClock * 1e6
chanBW = BW / nChan

(Status, Output) = commands.getstatusoutput("python2.5"                                                      \
                                            + " " + dirVEGAS + "/python/vegas_set_params.py"                 \
                                            + " -U --nogbt"                                                  \
                                            + " --host=" + ROACH_10GbE_IP                                    \
                                            + " --port=" + Port                                              \
                                            + " --packets=SPEAD"                                             \
                                            + " --npol=2"                                                    \
                                            + " --nchan=" + str(nChan)                                       \
                                            + " --chan_bw=" + str(chanBW)                                    \
                                            + " --nsubband=1"                                                \
                                            + " --sub0freq=2.4e9"                                            \
                                            + " --exposure=" + str(totInt)                                   \
                                            + " --fpgaclk=" + str(fpgaClock)                                 \
                                            + " --efsampfr=1.2e9"                                            \
                                            + " --hwexposr=0.5e-3"                                           \
                                            + " --obsmode=HBW")
if (0 != Status):
    sys.stderr.write("ERROR: vegas_set_params.py failed. " + Output + "\n")
    sys.exit(1)

