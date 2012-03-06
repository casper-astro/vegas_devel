import corr,time

roach = 'roach03'

## IPs at Greenbank
#dest_ip = 192*(2**24)+168*(2**16)+3*(2**8)+15
#src_ip = 192*(2**24)+168*(2**16)+3*(2**8)+17

## IPs at BWRC
dest_ip = 10*(2**24)+145
src_ip = 10*(2**24)+4

dest_port = 60000

mac_base = (2 << 40) + (2<<32)
fabric_port = 60000
acc_len=1023

fpga=corr.katcp_wrapper.FpgaClient(roach,7147)
time.sleep(1)

#boffile='vegas01_4tap_1024ch_220_r8b_2012_Feb_02_1638.bof'

## fast 357MHz
#boffile='vegas01_4tap_1024ch_220_r8b_2012_Feb_03_1409.bof'

## slow 150MHz (with added snap blocks for testing)
boffile='vegas01_4tap_1024ch_220_r9a_2012_Feb_13_2153.bof'

## 354 MHZ /w ADC snap
#boffile='vegas01_8b_snaps_2012_Feb_19_1129.bof'

# Unprogram the device
fpga.progdev('')
time.sleep(2)

# Program the Device
fpga.progdev(boffile)

# Set 10GbE NIC IP and Port
time.sleep(3)
fpga.tap_start('tap0','gbe0',mac_base+src_ip,src_ip,fabric_port)

time.sleep(3)

# Set destination IP
fpga.write_int('dest_ip',dest_ip)

# Set destination port
fpga.write_int('dest_port',dest_port)

# Set accumulation length
fpga.write_int('acc_len',acc_len)
time.sleep(1)

# Set sync period

#fftshift will be hardcoded for final design
fpga.write_int('fftshift',0b10101010101)

fpga.write_int('sg_period',2*16*1024*1024/8 -2)
fpga.write_int('sg_sync',0x12)

fpga.write_int('arm',0)
fpga.write_int('arm',1)
fpga.write_int('arm',0)

time.sleep(1)

fpga.write_int('sg_sync',0x13)

def reset():
    fpga.write_int('sg_sync',0x12) 
    fpga.write_int('arm',0)
    fpga.write_int('arm',1)
    fpga.write_int('arm',0)
    fpga.write_int('sg_sync',0x12)
    fpga.write_int('sg_sync',0x11)

reset()

