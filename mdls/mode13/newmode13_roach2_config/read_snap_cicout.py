#!/usr/bin/python2.6


import numpy, corr, time, struct, sys, logging, pylab, matplotlib, scipy

katcp_port = 7147
roach = '192.168.40.67'
fpga = corr.katcp_wrapper.FpgaClient(roach, katcp_port)

time.sleep(1)

fpga.write_int('snap2_ctrl', 0)
time.sleep(1)
fpga.write_int('snap2_ctrl', 7)
#fpga.write_int('trig', 0)
time.sleep(1)
#fpga.write_int('trig', 1)
#time.sleep(1)
#fpga.write_int('trig', 0)

d_0 = struct.unpack('>65536l', fpga.read('snap2_bram', 65536*4))

pylab.plot(d_0, label='data')
pylab.show()

