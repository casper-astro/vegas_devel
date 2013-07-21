#! /usr/bin/env python2.6

import corr,time,numpy,struct,sys
execfile('mixercic_funcs.py')
n_inputs = 4 # number of simultaneous inputs - should be 4 for final design
#lo_f = 0  # LO in MHz
lo_f = [104,104,104,104,104,104,104,104,104]
#boffile='v13_16r128dr_ver104_2013_Feb_19_2001.bof'
#boffile='v13_16r128dr_ver104_2013_Mar_05_1659.bof'
#boffile='v13_16r128dr_ver111_2013_Mar_06_1933.bof'
boffile = 'l8_ver115_2013_May_16_2052.bof'
roach = '192.168.40.82'

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


#lo_setup(fpga, lo_f, bandwidth=400, n_inputs, cnt_r_name='mixer_cnt', mixer_name='s1', bramlength=8)

lo_setup(fpga, lo_f[0], 1500, n_inputs, 'mixer_cnt', 's0', 8)
lo_setup(fpga, lo_f[1], 1500, n_inputs, 'mixer_cnt', 's1', 8)
lo_setup(fpga, lo_f[2], 1500, n_inputs, 'mixer_cnt', 's2', 8)
lo_setup(fpga, lo_f[3], 1500, n_inputs, 'mixer_cnt', 's3', 8)
lo_setup(fpga, lo_f[4], 1500, n_inputs, 'mixer_cnt', 's4', 8)
lo_setup(fpga, lo_f[5], 1500, n_inputs, 'mixer_cnt', 's5', 8)
lo_setup(fpga, lo_f[6], 1500, n_inputs, 'mixer_cnt', 's6', 8)
lo_setup(fpga, lo_f[7], 1500, n_inputs, 'mixer_cnt', 's7', 8)

time.sleep(1)

fpga.write_int('s0_quant_gain',2**20)
fpga.write_int('s1_quant_gain',2**20)
fpga.write_int('s2_quant_gain',2**20)
fpga.write_int('s3_quant_gain',2**20)
fpga.write_int('s4_quant_gain',2**20)
fpga.write_int('s5_quant_gain',2**20)
fpga.write_int('s6_quant_gain',2**20)
fpga.write_int('s7_quant_gain',2**20)

