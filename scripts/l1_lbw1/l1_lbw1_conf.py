#! /usr/bin/env python2.6

from subprocess import Popen, PIPE, STDOUT
import corr,time,numpy,struct,sys
#n_inputs = 4 # number of simultaneous inputs - should be 4 for final design

#boffile='l1_ver115_2013_Jul_19_1113.bof'
#boffile='l1_ver106_2013_Jun_24_1324.bof'
boffile='l1_ver117_2013_Jul_20_1645.bof'

roach = '192.168.40.99'
fpga = corr.katcp_wrapper.FpgaClient(roach)
mix_frac = 4./16. # mixing by 4/16 of adc clock freq (sampling rate)
dec_rate = 8. # decimation rate
nchan = 8192. # number of points in fft
brd_clk = 1500. # clocking frequency for ADCs

dest_ip  = 10*(2**24) +  145 #10.0.0.145
src_ip   = 10*(2**24) + 4  #10.0.0.4
dest_port     = 60000
fabric_port     = 60000
mac_base = (2<<40) + (2<<32)

def runbash(shell_command):
  event = Popen(shell_command, shell=True, stdin=PIPE, stdout=PIPE, stderr=STDOUT)
  return event.communicate()

def setfreq(freq):
  b=runbash('sg_ctrl freq '+str(freq))
  time.sleep(.5)
  return b[0].split(' ')[0]

def setampl(ampl):
  a=runbash('sg_ctrl ampl '+str(ampl))
  time.sleep(.5)
  b=runbash('sg_ctrl ampl')
  time.sleep(.5)
  return b[0].split(' ')[0]

def setmodoff():
  a=runbash('sg_ctrl mod off')
  time.sleep(.5)
  b=runbash('sg_ctrl mod')
  time.sleep(.5)
  return b[0].split(' ')[0]

def getadc0():
  adc0=np.fromstring(fpga.snapshot_get('adcsnap0',man_trig=True,man_valid=True)['data'],dtype='<i1')
  return adc0

def getadc1():
  adc1=np.fromstring(fpga.snapshot_get('adcsnap1',man_trig=True,man_valid=True)['data'],dtype='<i1')
  return adc1

def getfilter0_nogain():
  '''get snap block after pol0 of the dec_fir'''
  filter=np.fromstring(fpga.snapshot_get('filtersnap0_nogain',man_trig=True,man_valid=True)['data'],dtype='>i4').astype('float').view('complex')
  return filter

def getfilter1_nogain():
  '''get snap block after pol1 of the dec_fir'''
  filter=np.fromstring(fpga.snapshot_get('filtersnap1_nogain',man_trig=True,man_valid=True)['data'],dtype='>i4').astype('float').view('complex')
  return filter

def getfilter0():
  '''get snap block after pol0 of the dec_fir and bit selection'''
  filter=np.fromstring(fpga.snapshot_get('filtersnap0',man_trig=True,man_valid=True)['data'],dtype='>i1')
  return filter

def getfilter1():
  '''get snap block after pol1 of the dec_fir and bit selection'''
  filter=np.fromstring(fpga.snapshot_get('filtersnap1',man_trig=True,man_valid=True)['data'],dtype='<i1')
  return filter

def getrshp0():
  '''get snap block after the reshaper'''
  #select every other 2 pairs (real, imag) for 1 pol
  rshp=np.fromstring(fpga.snapshot_get('rshpsnap',man_trig=True)['data'],dtype='<i1')
  b=rshp[::4]
  c=rshp[1::4]
  #interleaving
  d=np.vstack((b,c)).reshape((-1),order='F').astype('float').view('complex')
  return d

def getrshp1():
  '''get snap block after the reshaper'''
  #select every other 2 pairs (real, imag) for 1 pol
  rshp=np.fromstring(fpga.snapshot_get('rshpsnap',man_trig=True)['data'],dtype='<i1')
  b=rshp[2::4]
  c=rshp[3::4]
  #interleaving
  d=np.vstack((b,c)).reshape((-1),order='F').astype('float').view('complex')
  return d

def plotrshp0():
  ''' band from 656.25 to 843.75'''
  d=getrshp0()
  #d=filter[::2].astype('float').view('complex')
  l=len(d)
  mixing_freq=(brd_clk*2)*mix_frac
  band_l=mixing_freq - (brd_clk/dec_rate)/2.
  bandwidth=brd_clk/dec_rate
  x=np.array(range(l))*bandwidth/l+band_l
  y=np.fft.fft(d)
  y=abs(np.fft.fftshift(y))
  plot(x,y)

  return y 

def filterresponse():
  resp=np.array([])
  freqs=np.array([])
  time.sleep(1)
  specs=abs(np.fft.fftshift(np.fft.fft(getrshp1())))

  mixing_freq=(brd_clk*2)*mix_frac
  bandwidth=brd_clk/dec_rate
  scan_range = 2
  freq_arr = np.array(range(int(nchan)*scan_range))*bandwidth/nchan + mixing_freq - bandwidth*scan_range*0.5
  
  for i in range(0,nchan*scan_range,201):
    freq=freq_arr[i]
    print "setting freq ",str(freq)
    setampl(-15)
    setfreq(freq)
    time.sleep(.3)
    print runbash('sg_ctrl freq')[0].split(' ')[0]
    resp_y=0
    for k in range(5):
      data=getrshp0()
      time.sleep(.3)
      y=np.fft.fft(data)
      y[0]=0
      y=abs(np.fft.fftshift(y))
      resp_y=resp_y+max(y)
    resp=np.append(resp, resp_y)
    freqs=np.append(freqs, freq)
    specs=np.vstack((specs, y))

  return resp, freqs, specs

def plotfilter0_nogain():
  ''' band from 656.25 to 843.75 when clocked at 187.5 MHz''' 
  filter=np.fromstring(fpga.snapshot_get('filtersnap0_nogain',man_trig=True,man_valid=True)['data'],dtype='>i4').astype('float').view('complex')
  l=len(filter)
  mixing_freq=(brd_clk*2)*mix_frac
  band_l=mixing_freq - (brd_clk/dec_rate)/2.
  bandwidth=brd_clk/dec_rate
  x=np.array(range(l))*bandwidth/(l-1)+band_l
  y=np.fft.fft(getfilter0_nogain())
  x=np.append(x[l/2:],x[:l/2])
  plot(x,y)

def plotfilter1_nogain():
  ''' band from 656.25 to 843.75 when clocked at 187.5 MHz''' 
  filter=np.fromstring(fpga.snapshot_get('filtersnap1_nogain',man_trig=True,man_valid=True)['data'],dtype='>i4').astype('float').view('complex')
  l=len(filter)
  mixing_freq=(brd_clk*2)*mix_frac
  band_l=mixing_freq - (brd_clk/dec_rate)/2.
  bandwidth=brd_clk/dec_rate
  x=np.array(range(l))*bandwidth/(l-1)+band_l
  y=np.fft.fft(getfilter1_nogain())
  x=np.append(x[l/2:],x[:l/2])
  plot(x,y)

def plotfilter0():
  ''' band from 656.25 to 843.75''' 
  filter = np.fromstring(fpga.snapshot_get('filtersnap0',man_trig=True,man_valid=True)['data'],dtype='<i1').astype('float').view('complex')
  l=len(filter)
  mixing_freq=(brd_clk*2)*mix_frac
  band_l=mixing_freq - (brd_clk/dec_rate)/2.
  bandwidth=brd_clk/dec_rate
  x=np.array(range(l))*bandwidth/(l-1)+band_l 
  y=np.fft.fft(filter)
  y=abs(np.fft.fftshift(y))
  #x=np.append(x[l/2:],x[:l/2])
  plot(x,y)

def plotfilter1():
  ''' band from 656.25 to 843.75''' 
  filter = np.fromstring(fpga.snapshot_get('filtersnap1',man_trig=True,man_valid=True)['data'],dtype='<i1').astype('float').view('complex')
  l=len(filter)
  mixing_freq=(brd_clk*2)*mix_frac
  band_l=mixing_freq - (brd_clk/dec_rate)/2.
  bandwidth=brd_clk/dec_rate
  x=np.array(range(l))*bandwidth/(l-1)+band_l
  y=np.fft.fft(filter)
  x=np.append(x[l/2:],x[:l/2])
  plot(x,y)

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

def plotpsd():
  data=getpacket()
  a=np.array(struct.unpack('<8208b', data), dtype=np.int8)[16:].astype('float').view('complex') 
  psd(a[::2].real,NFFT=1024,Fs=375)

def getpacketfft():
  data=getpacket()
  a=np.array(struct.unpack('<8208b', data), dtype=np.int8)[16:]
  b=a[::4]
  c=a[1::4]
  #interleave arrays b and c
  d=np.vstack((b,c)).reshape((-1),order='F').astype('float').view('complex')
  y=np.fft.fft(d)
  return y

def plotpacketfft():
  y=getpacketfft()
  l=len(y)
  x=x=np.array(range(l))*bandwidth/(l-1)+band_l
  x=np.append(x[l/2:],x[:l/2])
  plot(x,y)
  return d 

def setgain(gain):
  fpga.write_int('gain',gain)
 
def gbeconf():
  fpga.tap_start('tap0','gbe0',mac_base+src_ip,src_ip,fabric_port)
 
def main():
  print('Connecting to server %s on port... '%(roach)),
  time.sleep(2)
  if fpga.is_connected():
    print 'ok\n'	
  else:
    print 'ERROR\n'

  print '------------------------'
  print 'Programming FPGA with %s...' % boffile,
  fpga.progdev(boffile)
  print 'ok\n'
  time.sleep(2)

  print '------------------------'
  print 'Setting the port 0 linkup :',
  #time.sleep(1)
 # gbe0_link=bool(fpga.read_int('gbe0'))

  #print gbe0_link

  #if not gbe0_link:
  #  print 'There is no cable plugged into port0'

  print '------------------------'
  print 'Configuring receiver core...',   
#  gbeconf()
  fpga.tap_start('tap0','gbe0',mac_base+src_ip,src_ip,fabric_port)
  print 'done'

  print '------------------------'
  print 'Setting-up packet core...',

  print '------------------------'
  print 'Estimated board clock is %s...' % fpga.est_brd_clk()

  sys.stdout.flush()
  time.sleep(1)
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
  time.sleep(1) 

  # for -15dB coming in from the SG 
  setgain(2**30+2**29)
  print 'done'

if __name__ == "__main__":
  main()
