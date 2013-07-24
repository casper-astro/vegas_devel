#! /usr/bin/env python2.6

from subprocess import Popen, PIPE, STDOUT
import corr,time,numpy,struct,sys, socket
from matplotlib.pyplot import * 
execfile('mixercic_funcs.py')

n_inputs = 4 # number of simultaneous inputs - should be 4 for final design
bramlength = 10 # size of brams used in mixers (2^?)
lo_f = 91
lo_f_actual = lo_f
brd_clk = 1500. # bandwidth, in MHz, aka brd_clk
bw = brd_clk
dec_rate = 128. # decimation rate
#nchan = 8192. # number of points in fft

#boffile='v13_16r128dr_ver113b_2013_Mar_21_0034.bof'
#boffile='v13_16r64dr_ver114_2013_Apr_06_2235.bof'
#boffile='v13_16r128dr_ver117_2013_Apr_12_0131.bof'
#boffile='l8_ver115_2013_May_17_1027.bof'
#boffile='l8_ver117_2013_May_25_1242.bof'
boffile='l8_ver121_01_2013_Jul_21_1229.bof'

roach = '192.168.40.99'

dest_ip  = 10*(2**24) +  145 #10.0.0.145
src_ip   = 10*(2**24) + 4  #10.0.0.4

dest_port     = 60000
fabric_port     = 60000

mac_base = (2<<40) + (2<<32)


def runbash(shell_command):
  event = Popen(shell_command, shell=True, stdin=PIPE, stdout=PIPE, stderr=STDOUT)
  return event.communicate()

def setfreq(freq):
  b=runbash('./sg_ctrl freq '+str(freq))
  time.sleep(.5)
  return b[0].split(' ')[0]

def setampl(ampl):
  a=runbash('./sg_ctrl ampl '+str(ampl))
  time.sleep(.5)
  b=runbash('./sg_ctrl ampl')
  time.sleep(.5)
  return b[0].split(' ')[0]

def setmodoff():
  a=runbash('./sg_ctrl mod off')
  time.sleep(.5)
  b=runbash('./sg_ctrl mod')
  time.sleep(.5)
  return b[0].split(' ')[0]

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

def plotfft2(data, lo_f, bw, dec_rate):
    f_index = np.linspace(lo_f - bw/(1.*dec_rate), lo_f + bw/(1.*dec_rate), len(
data))
    plot(f_index, 10*np.log10(np.abs(np.fft.fftshift(np.fft.fft(data)))))


def plotfft(ax, data, lo_f, bw, dec_rate):
    f_index = np.linspace(lo_f - bw/(1.*dec_rate), lo_f + bw/(1.*dec_rate), len(
data))
    f_index = f_index - bw/(1.*dec_rate*len(data))
    ax.plot(f_index, 10*np.log10(np.abs(np.fft.fftshift(np.fft.fft(data)))))

def makecomplex(data0, data1):
    data_len = len(data0)
    complexdata = np.zeros(data_len, dtype=np.complex64)
    complexdata.real = data0
    complexdata.imag = data1
    return complexdata

def get_mixer_ri(pol, reim):
    mixer = np.fromstring(fpga.snapshot_get('s1_mixer_p'+str(pol)+reim,man_trig=True, man_valid=True)['data'],dtype='>i1')
    return mixer

def get_mixer(pol):
    mixer_re = get_mixer_ri(pol, 're')
    mixer_im = get_mixer_ri(pol, 'im')
    return mixer_re, mixer_im

def get_mixer_lmt():
    lmt = fpga.read_int('s1_mixer_cnt')
    return lmt

def setgain(subband, gain1, gain2):
  fpga.write_int('s'+str(subband)+'_quant_gain1',gain1)
  fpga.write_int('s'+str(subband)+'_quant_gain2',gain2)

def gbeconf():
  fpga.tap_start('tap0','gbe0',mac_base+src_ip,src_ip,fabric_port)

def getadc(i):
  adc=np.fromstring(fpga.snapshot_get('adcsnap'+str(i),man_trig=True,man_valid=True)['data'],dtype='<i1')
  return adc

def get_dr16(pol):
    '''
    subband 1, first CIC filter (dec_rate=16)
    '''
    dr16=np.fromstring(fpga.snapshot_get('s1_p'+str(pol)+'_dr16', man_trig=True,
 man_valid=True)['data'], dtype='>i8')
    dr16_re = dr16[0::2]
    dr16_im = dr16[1::2]
    dr16_out = makecomplex(dr16_re, dr16_im)
    return dr16_out

def get_1stcic(pol):
    first_cic = np.fromstring(fpga.snapshot_get('s1_p'+str(pol)+'_firstcic', man_trig=True, man_valid=True)['data'],dtype='>i4')
    first_cic_full = np.fromstring(fpga.snapshot_get('s1_p'+str(pol)+'_firstcic_full', man_trig=True, man_valid=True)['data'],dtype='>i8')
    fst_c_re = first_cic[0::4]
    fst_c_im = first_cic[1::4]
    fst_c_full_re = first_cic_full[0::4]
    fst_c_full_im = first_cic_full[1::4]
    first_cic_out = makecomplex(fst_c_re, fst_c_im)
    first_cic_full_out = makecomplex(fst_c_full_re, fst_c_full_im)
    return first_cic_out, first_cic_full_out

def get_halfband(pol):
    halfband = np.fromstring(fpga.snapshot_get('s1_p'+str(pol)+'_halfband', man_trig=True, man_valid=True)['data'], dtype='>i4')
    hb_re = halfband[0::4]
    hb_im = halfband[1::4]
    hb_out = makecomplex(hb_re, hb_im)
    return hb_out

def get_subband1(pol):
  ''' 
  This is for the output of the first subband
  '''
  s1p_raw = np.fromstring(fpga.snapshot_get('s1p'+str(pol)+'_snap',man_trig=True,man_valid=True)['data'],dtype='<i1')
  s1p_re = s1p_raw[0::2*(dec_rate/(2**n_inputs))]
  s1p_im = s1p_raw[1::2*(dec_rate/(2**n_inputs))]
  s1p = makecomplex(s1p_re, s1p_im)
  return s1p

def get_chc(pol):
  ''' 
  This is for the output of the first subband
  '''
  chc_raw = np.fromstring(fpga.snapshot_get('cic'+str(pol)+'_snap', man_trig=True, man_valid=True)['data'], dtype='>i4')
  chc_re = chc_raw[0::2*(dec_rate/(2**n_inputs))]
  chc_im = chc_raw[1::2*(dec_rate/(2**n_inputs))]
  chc = makecomplex(chc_re, chc_im)
  return chc

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
  f_index_x = np.linspace(lo_f_actual[i] - brd_clk/(1.*128), lo_f_actual[i] + brd_clk/(1.*128), l)
  subplot(211)
  plot(f_index_x, 10*np.log10(np.abs(np.fft.fftshift(np.fft.fft(X)))))
  f_index_y = np.linspace(lo_f_actual[i] - brd_clk/(1.*128), lo_f_actual[i] + brd_clk/(1.*128), l_y)
  subplot(212)
  plot(f_index_y, 10*np.log10(np.abs(np.fft.fftshift(np.fft.fft(Y)))))

def plot_dr16(lo_f):
    dr16_p1 = get_dr16(1)
    dr16_p2 = get_dr16(2)
    f, (ax1, ax2, ax3, ax4) = subplots(4)
    ax1.plot(dr16_p1.real, '-o')
    ax2.plot(dr16_p1.imag, '-o')
    ax3.plot(dr16_p2.real, '-o')
    ax4.plot(dr16_p2.imag, '-o')
    show()
    f, (ax1, ax2) = subplots(2)
    plotfft(ax1, dr16_p1, lo_f, bw, 16.)
    plotfft(ax2, dr16_p2, lo_f, bw, 16.)
    show()

def plot_1stcic(lo_f):
    c_p1, c_p1_full = get_1stcic_ri(1)
    c_p2, c_p2_full = get_1stcic_ri(2)
    f, (ax1, ax2) = subplots(2)
    plotfft(ax1, c_p1, lo_f, bw, 32.)
    plotfft(ax2, c_p2, lo_f, bw, 32.)
    show()
    f, (ax1, ax2) = subplots(2)
    plotfft(ax1, c_p1_full, lo_f, bw, 32.)
    plotfft(ax2, c_p2_full, lo_f, bw, 32.)
    show()

def plot_hb(lo_f):
    hb_p1 = get_halfband(1)
    hb_p2 = get_halfband(2)
    f, (ax1, ax2, ax3, ax4) = subplots(4)
    ax1.plot(hb_p1.real, '-o')
    ax2.plot(hb_p1.imag, '-o')
    ax3.plot(hb_p2.real, '-o')
    ax4.plot(hb_p2.imag, '-o')
    show()
    f, (ax1, ax2) = subplots(2)
    plotfft(ax1, hb_p1, lo_f, bw, 32.)
    plotfft(ax2, hb_p2, lo_f, bw, 32.)
    show()

def plot_chc(lo_f):
    chc_p1 = get_chc(1)
    chc_p2 = get_chc(2)
    f, (ax1, ax2, ax3, ax4) = subplots(4)
    ax1.plot(chc_p1.real, '-o')
    ax2.plot(chc_p1.imag, '-o')
    ax3.plot(chc_p2.real, '-o')
    ax4.plot(chc_p2.imag, '-o')
    show()
    f, (ax1, ax2) = subplots(2)
    plotfft(ax1, chc_p1, lo_f, bw, dec_rate)
    plotfft(ax2, chc_p2, lo_f, bw, dec_rate)
    show()

def lo_adjust(i, new_lo):
  lo_actual, lo_wave = lo_setup(fpga, new_lo, bw, n_inputs, 's'+str(i), bramlength, 1)
  lo_f_actual[i] = new_lo
  return lo_actual, lo_wave

def filterresponse(pol, lo_f, scan_range=1, skip=50):
  resp=np.array([])
  freqs=np.array([])
  time.sleep(1)
  data=get_subband1(pol)
  nchan=1.*len(data)
  specs=abs(np.fft.fftshift(np.fft.fft(data)))

  bandwidth=brd_clk*2./dec_rate
  freq_arr = np.array(range(int(nchan)*scan_range))*bandwidth/nchan + lo_f - bandwidth*scan_range*0.5
  
  for i in range(0,nchan*scan_range,skip):
    freq=freq_arr[i]
    print "setting freq ",str(freq)
    setampl(-15)
    setfreq(freq)
    time.sleep(.3)
    print runbash('./sg_ctrl freq')[0].split(' ')[0]
    resp_y=0
    for k in range(5):
      data=get_subband1(pol)
      adc=getadc(pol-1)
      time.sleep(.3)
      y=np.fft.fft(data)
      y0=np.fft.fft(adc)
      y[0]=0
      y0[0]=0
      y=abs(np.fft.fftshift(y))
      y0=abs(np.fft.fftshift(y0))
      resp_tmp = max(y)/max(y0)
      #resp_tmp = y[i]/max(y0) 
      resp_y=resp_y+max(y)
    resp=np.append(resp, resp_y)
    freqs=np.append(freqs, freq)
    specs=np.vstack((specs, y))

  return resp, freqs, specs

#-------------------------------------------------------
#
#-------------------------------------------------------

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

fpga.write_int('mode_sel', 1)
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

t_sleep = 1
lo_f_actual, wave = lo_setup(fpga, lo_f, brd_clk, n_inputs, 's1', bramlength, t_sleep)

time.sleep(1)

setgain(1, 2**12, 2**14)

print "Board Clock: ",fpga.est_brd_clk()
