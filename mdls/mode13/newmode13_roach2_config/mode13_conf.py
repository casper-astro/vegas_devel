#!/usr/bin/python2.6

import corr,time,numpy,struct,sys

bitstream = 'newmode13_2012_Aug_03_1511.bof'
#roach     = 'roach03'
roach     = '192.168.40.67'

#Decide where we're going to send the data, and from which addresses:
#dest_ip  = 10*(2**24) + 145 #10.0.0.145
dest_ip  = 10*(2**24) + 1*(2**8) + 146 #10.0.1.146
port     = 60000
#src_ip   = 10*(2**24) + 10  #10.0.0.10
src_ip   = 10*(2**24) + 1*(2**8) + 10  #10.0.1.10
mac_base = (2<<40) + (2<<32)
gbe0     = 'gbe0';

print('Connecting to server %s on port... '%(roach)),
fpga = corr.katcp_wrapper.FpgaClient(roach)
time.sleep(2)

if fpga.is_connected():
	print 'ok\n'	
else:
    print 'ERROR\n'

print '------------------------'
print 'Programming FPGA with %s...' %bitstream,
fpga.progdev(bitstream)
print 'ok\n'
time.sleep(5)

print '------------------------'
print 'Setting the port 0 linkup :',
#fpga.listdev()
gbe0_link=bool(fpga.read_int(gbe0))
print gbe0_link
if not gbe0_link:
   print 'There is no cable plugged into port0'
print '------------------------'
print 'Configuring receiver core...',   
# have to do this manually for now
fpga.tap_start('tap0',gbe0,mac_base+src_ip,src_ip,port)
print 'done'

print '------------------------'
print 'Setting-up packet core...',
sys.stdout.flush()
fpga.write_int('dest_ip',dest_ip)
fpga.write_int('dest_port',port)

fpga.write_int('sg_sync', 0b10100)
time.sleep(1)
fpga.write_int('arm', 0)
fpga.write_int('arm', 1)
fpga.write_int('arm', 0)
time.sleep(1)
fpga.write_int('sg_sync', 0b10101)
fpga.write_int('sg_sync', 0b10100)
print 'done'

