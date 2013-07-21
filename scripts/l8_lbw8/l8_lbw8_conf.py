#! /usr/bin/env python2.6

import corr,time,numpy,struct,sys, socket
from matplotlib.pyplot import * 
execfile('mixercic_funcs.py')

n_inputs = 4 # number of simultaneous inputs - should be 4 for final design
bramlength = 8 # size of brams used in mixers (2^?)
#lo_f = [104, 103, 75, 91, 92, 122, 135, 144]
lo_f = [75, 91, 92, 100, 104, 122, 135, 144]
lo_f_actual = lo_f
bw = 1200 # bandwidth, in MHz

#boffile='v13_16r128dr_ver113b_2013_Mar_21_0034.bof'
#boffile='v13_16r64dr_ver114_2013_Apr_06_2235.bof'
#boffile='v13_16r128dr_ver117_2013_Apr_12_0131.bof'
#boffile='l8_ver115_2013_May_17_1027.bof'
boffile='l8_ver117_2013_May_25_1242.bof'

roach = '192.168.40.82'

dest_ip  = 10*(2**24) +  145 #10.0.0.145
src_ip   = 10*(2**24) + 4  #10.0.0.4

dest_port     = 60000
fabric_port     = 60000

mac_base = (2<<40) + (2<<32)

def boardreset():
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

def setgain(gain):
  fpga.write_int('gain',gain)

def gbeconf():
  fpga.tap_start('tap0','gbe0',mac_base+src_ip,src_ip,fabric_port)

def getadc0():
  adc0=np.fromstring(fpga.snapshot_get('adcsnap0',man_trig=True,man_valid=True)['data'],dtype='<i1')
  return adc0

def getsubband1():
  ''' This is a plot after the first subband (pol1, real)'''
  s1p1re=np.fromstring(fpga.snapshot_get('s1p1re_snap',man_trig=True,man_valid=True)['data'],dtype='<i1')[::4]
  return s1p1re 

def getpacket():
  size=8208
  udp_ip='10.0.0.145'
  udp_port=60000
  sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
  sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
  sock.bind((udp_ip, udp_port))
  data, addr = sock.recvfrom(size)
  sock.close()
  return data

def plotpacketfft(i):
  data=getpacket()
  a=np.array(struct.unpack('>8208b', data), dtype=np.int8)[16:] #question here, >, or <?
  realX = a[0+i*4::32]
  imagX = a[1+i*4::32]
  realY = a[2+i*4::32]
  imagY = a[3+i*4::32]
  X = np.zeros(256, dtype=np.complex64)
  X.real = realX.astype(np.float)
  X.imag = imagX.astype(np.float)
  l = len(X)
  Y = np.zeros(256, dtype=np.complex64)
  Y.real = realY.astype(np.float)
  Y.imag = imagY.astype(np.float)
  l_y = len(Y)
  f_index_x = np.linspace(lo_f_actual[i] - bw/(1.*128), lo_f_actual[i] + bw/(1.*128), l)
  subplot(211)
  plot(f_index_x, 10*np.log10(np.abs(np.fft.fftshift(np.fft.fft(X)))))
  f_index_y = np.linspace(lo_f_actual[i] - bw/(1.*128), lo_f_actual[i] + bw/(1.*128), l_y)
  subplot(212)
  plot(f_index_y, 10*np.log10(np.abs(np.fft.fftshift(np.fft.fft(Y)))))

def plotsubband1fft():
  s1p1re = getsubband1()
  f_index = np.linspace(lo_f_actual[1] - bw/(2.*128), lo_f_actual[1] + bw/(2.*128), len(s1p1re))
  plot(f_index, 10*np.log10(np.abs(np.fft.fftshift(np.fft.fft(s1p1re)))))

def lo_adjust(i, new_lo):
  lo_setup(fpga, new_lo, bw, n_inputs, 's'+str(i), bramlength)
  lo_f_actual[i] = new_lo


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

fpga.write_int('mode_sel', 0)
time.sleep(1)
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


lo_f_actual[0] = lo_setup(fpga, lo_f[0], bw, n_inputs, 's0', bramlength)
lo_f_actual[1] = lo_setup(fpga, lo_f[1], bw, n_inputs, 's1', bramlength)
lo_f_actual[2] = lo_setup(fpga, lo_f[2], bw, n_inputs, 's2', bramlength)
lo_f_actual[3] = lo_setup(fpga, lo_f[3], bw, n_inputs, 's3', bramlength)
lo_f_actual[4] = lo_setup(fpga, lo_f[4], bw, n_inputs, 's4', bramlength)
lo_f_actual[5] = lo_setup(fpga, lo_f[5], bw, n_inputs, 's5', bramlength)
lo_f_actual[6] = lo_setup(fpga, lo_f[6], bw, n_inputs, 's6', bramlength)
lo_f_actual[7] = lo_setup(fpga, lo_f[7], bw, n_inputs, 's7', bramlength)

time.sleep(1)

fpga.write_int('s0_quant_gain',2**20)
fpga.write_int('s1_quant_gain',2**20)
fpga.write_int('s2_quant_gain',2**20)
fpga.write_int('s3_quant_gain',2**20)
fpga.write_int('s4_quant_gain',2**20)
fpga.write_int('s5_quant_gain',2**20)
fpga.write_int('s6_quant_gain',2**20)
fpga.write_int('s7_quant_gain',2**20)

print "Board Clock: ",fpga.est_brd_clk()
