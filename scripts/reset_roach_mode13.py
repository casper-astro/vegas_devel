import corr,time

roach = 'roach03'

fpga=corr.katcp_wrapper.FpgaClient(roach,7147)
time.sleep(1)

#rst counter
fpga.write_int('cnt_rst',1)
fpga.write_int('cnt_rst',0)


