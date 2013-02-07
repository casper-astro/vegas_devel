#! /opt/vegas/bin/python2.7

import corr,time,struct
roach = '192.168.40.80'

## IPs at Greenbank
#dest_ip = 192*(2**24)+168*(2**16)+3*(2**8)+15
#src_ip = 192*(2**24)+168*(2**16)+3*(2**8)+17

## IPs at BWRC
dest_ip = 10*(2**24)+145
src_ip = 10*(2**24)+4

dest_port = 60000

mac_base = (2 << 40) + (2<<32)
fabric_port = 60000

#acc_len=1023
acc_len=767
#acc_len=2046

fpga=corr.katcp_wrapper.FpgaClient(roach,7147)
time.sleep(1)

####
bram_name='s1_lo_3_lo_ram'
spectick = 1e-3
duration = []
duration.append(10e-3/spectick) #Blank duaration  
duration.append(100e-3/spectick) #cal duaration  

sstate = []

sstate.append(int(duration[0]) << 5 | 0b00111)
sstate.append(int(duration[1]) << 5 | 0b00110)
sstate.append(int(duration[0]) << 5 | 0b00101)
sstate.append(int(duration[1]) << 5 | 0b00100)
sstate.append(int(duration[0]) << 5 | 0b00011)
sstate.append(int(duration[1]) << 5 | 0b00010)
sstate.append(int(duration[0]) << 5 | 0b00001)
sstate.append(int(duration[1]) << 5 | 0b00000)

length = len(sstate)

fpga.write(bram_name,struct.pack('>'+str(length)+'I',*sstate))

print struct.unpack('>'+str(8)+'I',fpga.read(bram_name,length*4))
print sstate



