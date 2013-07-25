#! /usr/bin/env python2.6

from subprocess import Popen, PIPE, STDOUT
import corr,time,numpy,struct,sys, socket
from matplotlib.pyplot import * 
execfile('mixercic_funcs.py')
execfile('l8_debug.py')

n_inputs = 4 # number of simultaneous inputs - should be 4 for final design
bramlength = 10 # size of brams used in mixers (2^?)

mode_sel = 0  # 0 for mode l8_lbw8
t_sleep = 1
lo_f = 91
lo_f_actual = lo_f
brd_clk = 1500. # bandwidth, in MHz, aka brd_clk
bw = brd_clk
dec_rate = 128. # decimation rate
#nchan = 8192. # number of points in fft

#boffile='v13_16r128dr_ver113b_2013_Mar_21_0034.bof'
#boffile='v13_16r64dr_ver114_2013_Apr_06_2235.bof'
#boffile='v13_16r128dr_ver117_2013_Apr_12_0131.bof'
#boffile='l8_ver115_2013_May_17_1027.bof'
#boffile='l8_ver117_2013_May_25_1242.bof'
boffile='l8_ver121_01_2013_Jul_21_1229.bof'

roach = '192.168.40.99'

dest_ip  = 10*(2**24) +  145 #10.0.0.145
src_ip   = 10*(2**24) + 4  #10.0.0.4

dest_port     = 60000
fabric_port     = 60000

mac_base = (2<<40) + (2<<32)

#-------------------------------------------------------
#
#-------------------------------------------------------

print('Connecting to server %s on port... '%(roach)),
fpga = corr.katcp_wrapper.FpgaClient(roach)
time.sleep(2)

if fpga.is_connected():
	print 'ok\n'	
else:
    print 'ERROR\n'

print '------------------------'
print 'Programming FPGA with %s...' % boffile,
fpga.progdev(boffile)
print 'ok\n'
time.sleep(5)

print '------------------------'
print 'Setting the port 0 linkup :',

gbe0_link=bool(fpga.read_int('gbe0'))
print gbe0_link

if not gbe0_link:
   print 'There is no cable plugged into port0'

print '------------------------'
print 'Configuring receiver core...',   
# have to do this manually for now
fpga.tap_start('tap0','gbe0',mac_base+src_ip,src_ip,fabric_port)
print 'done'

print '------------------------'
print 'Setting-up packet core...',
sys.stdout.flush()
fpga.write_int('dest_ip',dest_ip)
fpga.write_int('dest_port',dest_port)

set_mode(mode_sel)
time.sleep(1)
fpga.write_int('sg_sync', 0b10100)
time.sleep(1)
fpga.write_int('arm', 0)
fpga.write_int('arm', 1)
fpga.write_int('arm', 0)
time.sleep(1)
fpga.write_int('sg_sync', 0b10101)
fpga.write_int('sg_sync', 0b10100)
print 'done'

#########################################



lo_f_actual, wave = lo_setup(lo_f, 's1')

time.sleep(1)

setgain(1, 2**12, 2**14)

print "Board Clock: ",fpga.est_brd_clk()
