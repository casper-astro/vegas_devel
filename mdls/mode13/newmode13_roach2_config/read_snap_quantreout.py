#!/usr/bin/python2.6


##################################################################
# Script Information
# This script can be used as iADC(2GSps) board testing
# Connect the input I+ with a frequence such as 13MHz and clock
# as 200MHz
# 20120517@CASPER.Berkeley
##################################################################
import numpy, corr, time, struct, sys, logging, pylab, matplotlib, scipy

katcp_port = 7147
roach = '192.168.40.67'
fpga = corr.katcp_wrapper.FpgaClient(roach, katcp_port)

time.sleep(1)

fpga.write_int('snap1_ctrl', 0)
time.sleep(1)
fpga.write_int('snap1_ctrl', 7)
time.sleep(1)
#fpga.write_int('trig', 0)
#time.sleep(1)
#fpga.write_int('trig', 1)
#time.sleep(1)
#fpga.write_int('trig', 0)

d_0 = struct.unpack('>65536l', fpga.read('snap1_bram', 65536*4))
#d_0 = struct.unpack('>8192l', fpga.read('snap1_bram', 8192*4))

d = numpy.array(d_0)
d = d.astype(numpy.int8)
d = d[0::8]
#print d.shape

f = numpy.fft.fftshift(numpy.abs(numpy.fft.fft(d, 4096)))
ch = numpy.linspace(-6.25, 6.25, 4096)

pylab.subplot(2, 1, 1)
pylab.plot(d, label='data')
pylab.subplot(2, 1, 2)
pylab.plot(ch, 10*numpy.log10(f), label='spectrum')
pylab.show()

