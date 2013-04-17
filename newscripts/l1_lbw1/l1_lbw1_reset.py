#! /opt/vegas/bin/python2.7

import corr,time

roach = '192.168.40.80'

fpga=corr.katcp_wrapper.FpgaClient(roach,7147)
time.sleep(1)

#fpga.write_int('reset',1)
#fpga.write_int('reset',0)

fpga.write_int('sg_sync', 0b10100)
time.sleep(1)
fpga.write_int('arm', 0)
fpga.write_int('arm', 1)
fpga.write_int('arm', 0)
time.sleep(1)
fpga.write_int('sg_sync', 0b10101)
fpga.write_int('sg_sync', 0b10100)
