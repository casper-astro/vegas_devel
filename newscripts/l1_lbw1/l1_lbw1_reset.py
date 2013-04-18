#! /opt/vegas/bin/python2.7

import corr,time,sys

roach = '192.168.40.80'

dest_ip  = 10*(2**24) +  145 #10.0.0.145
src_ip   = 10*(2**24) + 4  #10.0.0.4
dest_port     = 60000
fabric_port     = 60000
mac_base = (2<<40) + (2<<32)


fpga=corr.katcp_wrapper.FpgaClient(roach,7147)
time.sleep(1)

fpga.tap_start('tap0','gbe0',mac_base+src_ip,src_ip,fabric_port)

#fpga.write_int('reset',1)
#fpga.write_int('reset',0)
sys.stdout.flush()

fpga.write_int('sg_sync', 0b10100)
time.sleep(1)
fpga.write_int('arm', 0)
fpga.write_int('arm', 1)
fpga.write_int('arm', 0)
time.sleep(1)
fpga.write_int('sg_sync', 0b10101)
fpga.write_int('sg_sync', 0b10100)
