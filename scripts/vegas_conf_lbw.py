#! /usr/bin/env python2.7

import corr,time,numpy,struct,sys
execfile('mixercic_funcs.py')

#boffile='v13_aa_rii_16r128dr_ver103_2012_Sep_28_1740.bof'
#boffile='v13_ver107_2013_Jan_08_0113.bof'
#boffile='v13_na_ri_8r64dr_ver101_2013_Jan_24_1701.bof'
#boffile='v13_8r64dr_ver101_2013_Feb_04_2012.bof'
boffile='v13_8r64dr_ver101_2013_Feb_05_1731.bof'


roach = '192.168.40.80'

dest_ip  = 10*(2**24) +  145 #10.0.0.145
src_ip   = 10*(2**24) + 4  #10.0.0.4

dest_port     = 60000
fabric_port     = 60000

mac_base = (2<<40) + (2<<32)

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
lo_f = 0  # LO in MHz
lo_setup(fpga, lo_f, bandwidth=400, n_inputs=3, cnt_r_name='mixer_cnt', mixer_name='s1', bramlength=8)
time.sleep(3)

set_gain_sg(fpga, 's1_quant_gain', '%c%c%c%c'%(0x00, 0x00, 0x01, 0x00))
print 'Setting gain to: 0x00, 0x00, 0x01, 0x00'
time.sleep(1)






