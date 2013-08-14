#! /opt/vegas/bin/python2.7

from subprocess import Popen, PIPE, STDOUT
import corr,time,struct 
import numpy as np
roach = '192.168.40.99'

## IPs at Greenbank
#dest_ip = 192*(2**24)+168*(2**16)+3*(2**8)+15
#src_ip = 192*(2**24)+168*(2**16)+3*(2**8)+17


#
bw = 1500.
nchan = 1024


## IPs at BWRC
dest_ip = 10*(2**24)+145
src_ip = 10*(2**24)+4

dest_port = 60000

mac_base = (2 << 40) + (2<<32)
fabric_port = 60000

acc_len=767

fpga=corr.katcp_wrapper.FpgaClient(roach,7147)
time.sleep(1)

#boffile='v01_16r4t11f_ver141_2013_Feb_19_1801.bof' # 1 subband - seems to work
#boffile='h1k_ver100_2013_Apr_17_1957.bof'
boffile='h1k_ver102_2013_Aug_06_1711.bof'

# Program the Device
fpga.progdev(boffile)
time.sleep(1)

# Set 10GbE NIC IP and Port
fpga.tap_start('tap0','gbe0',mac_base+src_ip,src_ip,fabric_port)

# Set destination IP
fpga.write_int('dest_ip',dest_ip)

# Set destination port
fpga.write_int('dest_port',dest_port)

# Set accumulation length
fpga.write_int('acc_len',acc_len)

# Set FFT shift schedule
fpga.write_int('fftshift', 0b1010101010)


# Set sync period
time.sleep(1)
fpga.write_int('sg_period',2*acc_len*7*32*128*32-4)
#fpga.write_int('sg_period',2*16*1024*1024/8 -2)

#fpga.write_int('sg_period',acc_len*32*7*256-4)

fpga.write_int('sg_sync',0x12)
fpga.write_int('arm',0)
fpga.write_int('arm',1)
fpga.write_int('arm',0)
fpga.write_int('sg_sync',0x13)

def reset():
    fpga.write_int('sg_sync',0x12) 
    fpga.write_int('arm',0)
    fpga.write_int('arm',1)
    fpga.write_int('arm',0)
    fpga.write_int('sg_sync',0x12)
    fpga.write_int('sg_sync',0x11)

def runbash(shell_command):
  """ run shell_command """
  event = Popen(shell_command, shell=True, stdin=PIPE, stdout=PIPE, stderr=STDOUT)
  return event.communicate()

def setfreq(freq):
  """ For Berkely BWRC setup: set frequency for signal generator """
  b=runbash('./sg_ctrl freq '+str(freq))
  time.sleep(.5)
  return b[0].split(' ')[0]

def setampl(ampl):
  """ For Berkely BWRC setup: set frequency for signal generator """
  a=runbash('./sg_ctrl ampl '+str(ampl))
  time.sleep(.5)
  b=runbash('./sg_ctrl ampl')
  time.sleep(.5)
  return b[0].split(' ')[0]

def setmodoff():
  """ For Berkely BWRC setup: set frequency for signal generator """
  a=runbash('./sg_ctrl mod off')
  time.sleep(.5)
  b=runbash('./sg_ctrl mod')
  time.sleep(.5)
  return b[0].split(' ')[0]

def channelshape(nbin, num, scan_range, acc):
  '''
  Parameters: 
    nbin: the N-th bin to scan
    num:  how many points to test inside a bin
    scan_range: how many times of binwidth to scan? we scan on a larger range on both side of the center bin
    acc: for each test frequency input, how many accumulation do we do to obtain a response data point?
  Retuns:
    powers0: an array; response of the (N-1)th bin
    powers1: an array; response of the Nth bin
    powers2: an array; response of the (N+1)th bin
    freqs:   an array; the test frequencies
    specs:   a 2-D array; the un-accumulated spectra; one spectrum for each test frequency
  Notes:
    (1) The unit for data in the powers0, powers1, powers2 arrays is dB/10
  '''
  powers0=np.array([])
  powers1=np.array([])
  powers2=np.array([])
  freqs=np.array([])

  deltaf = bw/nchan
  specs=getrshp()[:nchan]

  freqs1=np.array(range(int(num)*scan_range+1))*1./num*deltaf+nbin*deltaf-deltaf*(scan_range/2.)

  for freq in freqs1:
    print "setting freq ",str(freq)
    setfreq(freq)  # for Berkeley BWRC settings
    time.sleep(.5) 
    print runbash('./sg_ctrl freq')[0].split(' ')[0] #for Berkeley BWRC settings

    spec_bm1=0
    spec_bin=0
    spec_bp1=0

    for k in range(acc):
      spec=getrshp()[:nchan]
      spec_bm1=spec_bm1+spec[nbin-1]
      spec_bin=spec_bin+spec[nbin]
      spec_bp1=spec_bp1+spec[nbin+1]

    specs = np.vstack((specs, spec))
    freqs = np.append(freqs, freq)
    powers0=np.append(powers0, spec_bm1)
    powers1=np.append(powers1, spec_bin)
    powers2=np.append(powers2, spec_bp1)
    print 'Power at bin: ', str(spec_bin)
     
  return np.log10(powers0), np.log10(powers1), np.log10(powers2), freqs, specs

def getadc0():
  """ Read the output of adc0  """
  adc0=np.fromstring(fpga.snapshot_get('adcsnap0',man_trig=True,man_valid=True)['data'],dtype='<i1')
  return adc0

def getadc1():
  """ Read the output of adc1  """
  adc1=np.fromstring(fpga.snapshot_get('adcsnap0',man_trig=True,man_valid=True)['data'],dtype='<i1')
  return adc1

def getvacc():
  a=np.fromstring(fpga.snapshot_get('vaccout')['data'],dtype='>i4')[::4]
  return a[:32]

def getrshp():
  """ grabs a snapshot of XX values """
  rshp=np.fromstring(fpga.snapshot_get('rshpout')['data'],dtype='>i4')[::4]
  return rshp

def intrlv(ar1,ar2):
  """interleave two arrays """
  ar3 = np.zeros((len(ar1) + len(ar2)))
  ar3[0::2] = ar1
  ar3[1::2] = ar2
  return ar3 
       
def getstokes():
  stokes=np.fromstring(fpga.snapshot_get('stokesout',man_valid=True,man_trig=True)['data'],dtype='>i4')[::4]
  return stokes

def extract(reg,offset,bitwidth):
  reg = reg >> offset
  mask=1 
  for i in range(bitwidth-1):
    mask=(mask << 1) + 1
  return reg & mask

reset()

