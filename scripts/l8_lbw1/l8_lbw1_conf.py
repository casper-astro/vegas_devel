#! /usr/bin/env python2.6

import corr,time,numpy,struct,sys,socket
execfile('mixercic_funcs.py')
n_inputs = 4 # number of simultaneous inputs - should be 4 for final design
lo_f = 91  # LO in MHz
lo_f_actual = lo_f
bw = 1500 # Input siganl bandwidth, in MHz
#boffile='v81_16r128dr_ver101_2013_Apr_10_0126.bof'
boffile='l8_ver115_2013_May_17_1027.bof'
roach = '192.168.40.99'

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

def setgain(gain):
  fpga.write_int('gain',gain)

def gbeconf():
  fpga.tap_start('tap0','gbe0',mac_base+src_ip,src_ip,fabric_port)

def getadc0():
  adc0=np.fromstring(fpga.snapshot_get('adcsnap0',man_trig=True,man_valid=True)[
'data'],dtype='<i1')
  return adc0

def getsubband():
  ''' This is a plot after the first subband (pol1, real)'''
  s0p1re=np.fromstring(fpga.snapshot_get('s0p1re_snap',man_trig=True,man_valid=True)['data'],dtype='<i1')[::4]
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

def getpacketdata():
  data = getpacket()
  a = np.array(struct.unpack('>8208b', data), dtype=np.int8)[16:]
  realX = a[0::32].astype(np.float)
  imagX = a[1::32].astype(np.float)
  realY = a[2::32].astype(np.float)
  imagY = a[3::32].astype(np.float)
  return realX, imagX, realY, imagY

def plotpacketfft():
  data = getpacket()
  a=np.array(struct.unpack('>8208b', data), dtype=np.int8)[16:] #question here, >, or <?
  realX = a[0::32]
  imagX = a[1::32]
  realY = a[2::32]
  imagY = a[3::32]
  print size(realX.astype(np.float))
  X = np.zeros(2048, dtype=np.complex64)
  X.real = realX.astype(np.float)
  X.imag = imagX.astype(np.float)
  l = len(X)
  Y = np.zeros(2048, dtype=np.complex64)
  Y.real = realY.astype(np.float)
  Y.imag = imagY.astype(np.float)
  l_y = len(Y)
  f_index_x = np.linspace(lo_f_actual - bw/(2.*128), lo_f_actual + bw/(2.*
128), l)
  subplot(211)
  plot(f_index_x, 10*np.log10(np.abs(np.fft.fftshift(np.fft.fft(X)))))
  f_index_y = np.linspace(lo_f_actual - bw/(2.*128), lo_f_actual + bw/(2.*
128), l_y)
  subplot(212)
  plot(f_index_y, 10*np.log10(np.abs(np.fft.fftshift(np.fft.fft(Y)))))

def plotsubbandfft():
  s0p1re = getsubband()
  f_index = np.linspace(lo_f_actual - bw/(2.*128), lo_f_actual + bw/(2.*128), len(s1p1re))
  plot(f_index, 10*np.log10(np.abs(np.fft.fftshift(np.fft.fft(s1p1re)))))


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
print 'Setting the port 0 linkup :',2
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
time.sleep(1)

fpga.write_int('mode_sel', 1)
time.sleep(1)
fpga.write_int('sg_sync', 0b10100)
time.sleep(1)
fpga.write_int('arm', 0)
time.sleep(1)
fpga.write_int('arm', 1)
time.sleep(1)
fpga.write_int('arm', 0)
time.sleep(1)
fpga.write_int('sg_sync', 0b10101)
fpga.write_int('sg_sync', 0b10100)
print 'done'

#########################################


#lo_setup(fpga, lo_f, bandwidth=400, n_inputs, cnt_r_name='mixer_cnt', mixer_name='s1', bramlength=8)

lo_f_actual = lo_setup(fpga, lo_f, 1500, n_inputs, 's0', 8)

time.sleep(1)

fpga.write_int('s0_quant_gain',2**20)

