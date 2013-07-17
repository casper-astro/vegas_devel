#!/usr/bin/python2.6


import numpy, corr, time, struct, sys, logging, pylab, matplotlib, scipy

katcp_port = 7147
roach = '192.168.40.70'
fpga = corr.katcp_wrapper.FpgaClient('192.168.40.70', katcp_port)

time.sleep(1)

fpga.write_int('s32_cicp1re_ctrl', 0)
time.sleep(1)
fpga.write_int('s32_cicp1re_ctrl', 7)
#fpga.write_int('trig', 0)
time.sleep(1)
#fpga.write_int('trig', 1)
#time.sleep(1)
#fpga.write_int('trig', 0)

d_0 = struct.unpack('>65536l', fpga.read('s32_cicp1re_bram', 65536*4))

pylab.plot(d_0, label='data')
pylab.show()

