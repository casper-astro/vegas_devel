#! /usr/bin/env python

import sys
from binascii import hexlify
from guppi.client import Client
from optparse import OptionParser

# Parse command line, figure out what to do
par = OptionParser()
par.add_option("-q", "--query", dest="query",
        help="Query current HW settings (mode, clock)",
        action="store", default=None)
par.add_option("-m", "--mode", dest="mode",
        help="GUPPI mode to load",
        action="store", default=None)
par.add_option("-l", "--list", dest="list",
        help="List available modes",
        action="store_true", default=False)
par.add_option("-L", "--longlist", dest="longlist",
        help="List available modes with lots more info",
        action="store_true", default=False)
par.add_option("-c", "--clock", dest="clock",
        help="Sampling clock rate in MHz (200, 800)",
        action="store", default=None)
par.add_option("-f", "--force", dest="force",
        help="Force reload even if params match current settings",
        action="store_true", default=False)
(opt,arg) = par.parse_args()

# List of GUPPI modes (matching sets of bof files)
# TODO if load order matters we may want to change some of this

# BEE2 IP Addr settings
ip_text = 'begin\nmac = 10:10:10:10:10:11\nip = 192.168.3.8\n' + \
        'gateway = 192.168.3.8\nport = 50000\nend\n'

# Standard 2k mode
bofs_2k = [
        'BEE2/b2_GDSP_U1_4K_800_A_XA_fpga1_2008_Jul_30_1356.bof',
        'BEE2/b2_GDSP_U3_4K_800_A_XA_fpga3_2008_Jul_30_1414.bof',
        'BEE2/b2_GOUT_U2_4K_800_A_NR_fpga2_2008_Sep_15_1400.bof'
        ]
bofs_2k.sort()
init_2k = {
        'BEE2/FPGA1/FFT_SHIFT':'aaaaaaaa',
        'BEE2/FPGA1/LE_CNTRL':'00000000',
        'BEE2/FPGA1/SAMP_CMD':'00000000',
        'BEE2/FPGA1/DC_SAMP_EN':'00000001',
        'BEE2/FPGA1/DC_BINS_EN':'00000001',
        'BEE2/FPGA3/FFT_SHIFT':'aaaaaaaa',
        'BEE2/FPGA3/LE_CNTRL':'00000000',
        'BEE2/FPGA3/SAMP_CMD':'00000000',
        'BEE2/FPGA3/DC_SAMP_EN':'00000001',
        'BEE2/FPGA3/DC_BINS_EN':'00000001',
        'BEE2/FPGA2/GUPPi_PIPES_ARM':'00000000',
        'BEE2/FPGA2/OFFSET_I':'00000000',
        'BEE2/FPGA2/OFFSET_Q':'00000000',
        'BEE2/FPGA2/OFFSET_U':'00000000',
        'BEE2/FPGA2/OFFSET_V':'00000000',
        'BEE2/FPGA2/SCALE_I':'01000000',
        'BEE2/FPGA2/SCALE_Q':'01000000',
        'BEE2/FPGA2/SCALE_U':'01000000',
        'BEE2/FPGA2/SCALE_V':'01000000',
        'BEE2/FPGA2/ACC_LENGTH':'0000000f',
        'BEE2/FPGA2/DEST_IP':'c0a80307',
        'BEE2/FPGA2/DEST_PORT':'0000c350',
        'BEE2/FPGA2/DC_BINS_EN':'00000001',
        'BEE2/FPGA2/ten_GbE':hexlify(ip_text)
        }

# "1SFA" 4k mode
bofs_4k_1sfa = [
        'BEE2/bGDSP_U1_8K_248_A_00_fpga1_2008_Oct_22_1407.bof',
        'BEE2/bGDSP_U3_8K_248_A_00_fpga3_2008_Oct_22_1427.bof',
        'BEE2/bGOUT_1SFA_D16_fpga2_2009_Mar_27_1156.bof'
        ]
bofs_4k_1sfa.sort()
init_4k_1sfa = {
        'BEE2/FPGA1/FFT_SHIFT':'aaaaaaaa',
        'BEE2/FPGA1/LE_CNTRL':'00000000',
        'BEE2/FPGA1/SAMP_CMD':'00000000',
        'BEE2/FPGA1/DC_EN':'00000001',
        'BEE2/FPGA3/FFT_SHIFT':'aaaaaaaa',
        'BEE2/FPGA3/LE_CNTRL':'00000000',
        'BEE2/FPGA3/SAMP_CMD':'00000000',
        'BEE2/FPGA3/DC_EN':'00000001',
        'BEE2/FPGA2/GUPPi_PIPES_ARM':'00000000',
        'BEE2/FPGA2/OFFSET_I':'00000000',
        'BEE2/FPGA2/OFFSET_Q':'00000000',
        'BEE2/FPGA2/OFFSET_U':'00000000',
        'BEE2/FPGA2/OFFSET_V':'00000000',
        'BEE2/FPGA2/SCALE_I':'01000000',
        'BEE2/FPGA2/SCALE_Q':'01000000',
        'BEE2/FPGA2/SCALE_U':'01000000',
        'BEE2/FPGA2/SCALE_V':'01000000',
        'BEE2/FPGA2/ACC_LENGTH':'0000000f',
        'BEE2/FPGA2/DEST_IP':'c0a80307',
        'BEE2/FPGA2/DEST_PORT':'0000c350',
        'BEE2/FPGA2/DC_BINS_EN':'00000001',
        'BEE2/FPGA2/GUPPi_PIPES_BW_SEL':'00000002',
        'BEE2/FPGA2/BW_SEL':'00000001',
        'BEE2/FPGA2/ROL_SEL':'00000001',
        'BEE2/FPGA2/ten_GbE':hexlify(ip_text)
        }

modelist = {'2k':bofs_2k, '4k':bofs_4k_1sfa}
initlist = {'2k':init_2k, '4k':init_4k_1sfa}

# List modes if needed
if opt.list or opt.longlist:
    for m in sorted(modelist.keys()):
        if opt.longlist:
            print "Mode '%s'" % m
            print "  bofs:"
            for b in modelist[m]:
                print "    %s" % b
            print "  init defaults:"
            for p in sorted(initlist[m].items()):
                print "   ", p
            print
        else:
            print m
    sys.exit()

# Validate arguments
if opt.mode != None and opt.mode not in modelist.keys():
    print "Requested mode '%s' does not exist" % opt.mode
    sys.exit()

# Connect to guppi client
c = Client()

# Check clock rate
clock_current = c.get('SYNTH/CFRQ/VALUE')
clock_current = clock_current.strip("MHZmhz")
if opt.query == 'clock':
    print clock_current
    sys.exit()
elif opt.query==None:
    print "Current clock rate is %s" % clock_current

# Check currently loaded mode
bofs_current = c.unload()
bofs_current.sort()
if bofs_current in modelist.values():
    for (k,v) in modelist.iteritems():
        if bofs_current==v:
            mode_current = k
    if opt.query==None:
        print "Currently loaded mode is '%s'" % mode_current
else:
    if opt.query==None:
        print "Currently loaded mode is unrecognized:"
        for b in bofs_current:
            print "  %s" % b
    mode_current = 'unknown'

if opt.query == 'mode':
    print mode_current
    sys.exit()


# If new clock rate requested, load it
clock_changed = False
if opt.clock != None:
    opt.clock = opt.clock.strip("MHZmhz")
    if opt.clock != clock_current or opt.force:
        print "Changing clock rate to %s ..." % opt.clock,
        # TODO: reset procedure goes here
        # TODO: catch any errors
        print "ok"
        clock_changed = True
        clock_current = opt.clock
    else:
        print "No clock reset required"

# If new bofs required, load them
mode_changed = False
if opt.mode != None:
    if opt.mode != mode_current or opt.force or clock_changed:
        print "Loading '%s' mode ..." % opt.mode,
        # TODO: bof load procedure goes here
        # TODO: catch any errors
        print "ok"
        mode_changed = True
        mode_current = opt.mode
    else:
        print "No reload required"
else:
    if clock_changed:
        print "Reloading '%s' mode ..." % mode_current,
        # TODO: bof load here
        # TODO: catch any errors
        print "ok"
        mode_changed = True

# If bofs were reloaded, we need to re-init
if mode_changed:
    print "Initializing '%s' mode ..." % mode_current,
    # TODO: any clock-rate dependent settings
    # TODO: run init procedure
    # TODO: catch any errors
    # TODO: arm()
    print "ok"

