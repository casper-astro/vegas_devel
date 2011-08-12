import corr,time

roach = 'roach03'
dest_ip = 10*(2**24)+145
src_ip = 10*(2**24)+4
dest_port = 60000

mac_base = (2 << 40) + (2<<32)
fabric_port = 60000

acc_len=511
fft_size=2**5
simul_inputs=4
lcm=6
pfb_taps=4
sync_period = acc_len*lcm*pfb_taps*fft_size/simul_inputs

fpga=corr.katcp_wrapper.FpgaClient(roach,7147)
time.sleep(1)

# Set accumulation length
# fpga.write_int('acc_len',acc_len)
time.sleep(1)

# Print accumulation length
# print "Accumulation length = %d" % fpga.read_int('acc_len')

#rst counter
fpga.write_int('rst',1)
fpga.write_int('rst',0)


