import corr,time, struct, cmath, math

roach = 'roach03'
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

# boffile='mode13_fastnet_2011_Aug_18_2039.bof'
boffile='vegas13_fastnettest_250mhz_r1_2011_Aug_28_1712.bof'

# Unprogram the device
fpga.progdev('')
time.sleep(2)

# Program the Device
fpga.progdev(boffile)

# Set 10GbE NIC IP and Port
time.sleep(3)
fpga.tap_start('tap0','ten_GbE',mac_base+src_ip,src_ip,fabric_port)
time.sleep(3)

# Set destination IP
fpga.write_int('dest_ip',dest_ip)
fpga.write_int('ip_select',1)

# Set destination port
fpga.write_int('dest_port',dest_port)
fpga.write_int('port_select',1)

# Set sync period
fpga.write_int('sync_period',sync_period)
fpga.write_int('sync_sel',1)

# Create test waveform
wave = [50*cmath.exp(2j*math.pi*t/16) for t in range(16)]
wave_comb = []
map(wave_comb.extend, zip([(int)(w.real) for w in wave], [int(x.imag) for x in wave]))
fmt_string = "> " + "xxbb"*16
wave_packed = struct.pack(fmt_string, *wave_comb)

# Load test waveform into BRAM
fpga.write('test_waveform', wave_packed)

#rst 10 GBE core
fpga.write_int('eth_rst',1)
fpga.write_int('eth_rst',0)

#rst counter
fpga.write_int('cnt_rst',1)
fpga.write_int('cnt_rst',0)
