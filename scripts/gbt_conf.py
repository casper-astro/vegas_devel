import corr,time

roach = 'roach03'

## IPs at Greenbank
dest_ip = 192*(2**24)+168*(2**16)+3*(2**8)+15
src_ip = 192*(2**24)+168*(2**16)+3*(2**8)+17

#dest_ip = 10*(2**24)+145
#src_ip = 10*(2**24)+4

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

#boffile='mode01_full_2011_Aug_06_1914.bof'
#boffile='mode01_2011_Aug_07_1833.bof'
#
boffile='mode01_full_2011_Aug_14_1541.bof'

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
fpga.write_int('ip_select',1)

# Set destination port
fpga.write_int('dest_port',dest_port)
fpga.write_int('port_select',1)

# Set accumulation length
fpga.write_int('acc_len',acc_len)
time.sleep(1)

# Set sync period
fpga.write_int('sync_period',sync_period)
fpga.write_int('sync_period_sel',1)
fpga.write_int('rst',1)
fpga.write_int('rst',0)
fpga.write_int('sync_gen_sel',0)

#fpga.print_10gbe_core_details('gbe0',arp=True)

#rst counter
fpga.write_int('rst',1)
fpga.write_int('rst',0)


