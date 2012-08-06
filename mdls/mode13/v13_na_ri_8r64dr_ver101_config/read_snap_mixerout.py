#!/usr/bin/python2.6


##################################################################
# Script Information
# This script can be used as iADC(2GSps) board testing
# Connect the input I+ with a frequence such as 13MHz and clock
# as 200MHz
# 20120517@CASPER.Berkeley
##################################################################
import numpy, corr, time, struct, sys, logging, pylab, matplotlib, scipy

#connect roach
fpga = corr.katcp_wrapper.FpgaClient('192.168.40.70')
time.sleep(1)

#pay attention, the snap module is controled by the inner software register
#as snap64_ctrl which can be found at /proc/pid/hw/ioreg/
#the number write to the snap ctrl register is from 0 to 7, do not use the outer trig
#to control the snap module

fpga.write_int('s1_mixer_snap64_ctrl', 0)
time.sleep(1)
fpga.write_int('s1_mixer_snap64_ctrl', 7)
time.sleep(1)


#d_0 = struct.unpack('>8192b', fpga.read('s1_mixer_snap64_bram_msb', 2048*4))
d_0 = struct.unpack('>8192b', fpga.read('s1_mixer_snap64_bram_lsb', 2048*4))

fd_0 = []

#put the parrellel data into a serial array
for i in range(2048):
        
    fd_0.append(d_0[i*4+0]/128.0)
    fd_0.append(d_0[i*4+1]/128.0)
    fd_0.append(d_0[i*4+2]/128.0)
    fd_0.append(d_0[i*4+3]/128.0)


#plot it
pylab.plot(fd_0, label='data')
pylab.show()

