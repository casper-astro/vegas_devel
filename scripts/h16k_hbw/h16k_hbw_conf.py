#! /opt/vegas/bin/python2.7

import corr,time,struct
execfile('hbw_funcs.py')

 
roach = '192.168.40.99'

## IPs at Greenbank
#dest_ip = 192*(2**24)+168*(2**16)+3*(2**8)+15
#src_ip = 192*(2**24)+168*(2**16)+3*(2**8)+17

## IPs at BWRC
dest_ip = 10*(2**24)+145
src_ip = 10*(2**24)+4

dest_port = 60000

mac_base = (2 << 40) + (2<<32)
fabric_port = 60000

bw = 1500.
nchan = 1024*16.  #h16k

#acc_len=1023
acc_len=96 #h16k
#acc_len=2046
fftshift=0b101110111111111 #this is what the setting was in Apr4 commit 

fpga=corr.katcp_wrapper.FpgaClient(roach,7147)
time.sleep(1)

#boffile='newmode_2013_Mar_26_1953.bof'
#boffile='newmode_ver102_2013_Apr_04_1517.bof'
#boffile='h16k_ver100_2013_Apr_17_1725.bof'
#boffile='h16k_ver104_2013_Aug_06_1117.bof'   #<---MW
#boffile='h16k_ver102_2013_Aug_01_1254.bof'   #<---HC
#boffile='h16k_ver103_2013_Aug_03_2049.bof'   #<---HC
#boffile='h16k_ver104_2013_Aug_06_1117.bof'   #<---HC
#boffile='h16k_ver107_01_2013_Aug_11_1144.bof' #<---HC
#boffile='h16k_ver102_2013_Aug_19_1340.bof' #<--- HC ver107 compiled by MW
boffile='h16k_ver102_2013_Aug_20_0053.bof'

# Program the Device
fpga.progdev(boffile)
time.sleep(1)

# Set 10GbE NIC IP and Port
fpga.tap_start('tap0','gbe0',mac_base+src_ip,src_ip,fabric_port)

# Set destination IP
fpga.write_int('dest_ip',dest_ip)

# Set destination port
fpga.write_int('dest_port',dest_port)

# Set accumulation length
fpga.write_int('acc_len',acc_len)

# Set FFT shift schedule
#fpga.write_int('fftshift', 0b101110111111111)  #h16k, but where did I get this idea?
#fpga.write_int('fftshift', 0b1010101010)  # <--- this is the same as in mode h1k
fpga.write_int('fftshift', fftshift)

# Set sync period
time.sleep(1)
fpga.write_int('sg_period', acc_len*(2*11)*4*2048-4) #h16k
#fpga.write_int('sg_period',2*acc_len*7*32*128*32-4)
#fpga.write_int('sg_period',2*16*1024*1024/8 -2)

#fpga.write_int('sg_period',acc_len*32*7*256-4)

fpga.write_int('sg_sync',0x12)
fpga.write_int('arm',0)
fpga.write_int('arm',1)
fpga.write_int('arm',0)
fpga.write_int('sg_sync',0x13)

reset()
#debug_bram = getdebug()


#np.fromstring(u.snapshot_get('vaccout',man_valid=True,man_trig=True)['data'],dtype='>i4')

#np.fromstring(u.snapshot_get('rshpout')['data'],dtype='>i4')[::2]


