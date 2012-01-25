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

acc_len=255
fft_size=2**5
simul_inputs=4
lcm=6
pfb_taps=4
sync_period = (acc_len+1)*lcm*pfb_taps*fft_size/simul_inputs

fpga=corr.katcp_wrapper.FpgaClient(roach,7147)
time.sleep(1)

#
#boffile='vegas01_4tap_1024ch_220_r6c_2012_Jan_18_1626.bof'
#boffile='vegas01_4tap_1024ch_220_r6e_2012_Jan_21_1753.bof'
boffile='vegas01_4tap_1024ch_220_r6e_2012_Jan_23_1813.bof'

# Unprogram the device
fpga.progdev('')
time.sleep(2)

# Program the Device
fpga.progdev(boffile)

# Set 10GbE NIC IP and Port
time.sleep(3)
fpga.tap_start('tap0','gbe0',mac_base+src_ip,src_ip,fabric_port)
fpga.tap_start('tap0','gbe1',mac_base+src_ip+1,src_ip+1,fabric_port+1)
time.sleep(3)

# Set destination IP
fpga.write_int('dest_ip',dest_ip)
fpga.write_int('ip_select',1)

# Set destination port
fpga.write_int('dest_port',dest_port)
fpga.write_int('port_select',1)

# Set accumulation length
fpga.write_int('acc_len',acc_len)
time.sleep(1)

# Set sync period
#fpga.write_int('sync_period',sync_period)
#fpga.write_int('sync_period_sel',1)
#fpga.write_int('rst',1)
#fpga.write_int('rst',0)
#fpga.write_int('sync_gen_sel',0)

#fpga.print_10gbe_core_details('gbe0',arp=True)
fpga.write_int('fftshift',0x77777777)

fpga.write_int('sg_period',2**30)
fpga.write_int('sg_sync',0x12)

fpga.write_int('arm',0)
fpga.write_int('arm',1)
fpga.write_int('arm',0)

time.sleep(1)

fpga.write_int('rst',1)
fpga.write_int('rst',0)

fpga.write_int('sg_sync',0x13)


def reset():
    fpga.write_int('sg_sync',0x12) 
    
    #rst counter
    fpga.write_int('rst',1)
    fpga.write_int('rst',0)

    fpga.write_int('arm',0)
    fpga.write_int('arm',1)
    fpga.write_int('arm',0)
    fpga.write_int('sg_sync',0x12)
    fpga.write_int('sg_sync',0x11)


reset()

