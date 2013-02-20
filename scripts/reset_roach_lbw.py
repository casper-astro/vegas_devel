#! /opt/vegas/bin/python2.7

import corr,time

roach = '192.168.40.80'

fpga=corr.katcp_wrapper.FpgaClient(roach,7147)
time.sleep(1)

#rst counter
fpga.write_int('reset',1)
fpga.write_int('reset',0)

#fpga.write_int('cnt_rst',1)
#fpga.write_int('cnt_rst',0)


