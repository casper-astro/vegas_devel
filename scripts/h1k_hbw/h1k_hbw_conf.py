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
boffile='h1k_ver100_2013_Apr_17_1957.bof'

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
  event = Popen(shell_command, shell=True, stdin=PIPE, stdout=PIPE, stderr=STDOUT)
  return event.communicate()

def setfreq(freq):
  b=runbash('./sg_ctrl freq '+str(freq))
  time.sleep(.5)
  return b[0].split(' ')[0]

def setampl(ampl):
  a=runbash('sg_ctrl ampl '+str(ampl))
  time.sleep(.5)
  b=runbash('sg_ctrl ampl')
  time.sleep(.5)
  return b[0].split(' ')[0]

def channelshape(nbin, num, scan_range):
  powers0=np.array([])
  powers1=np.array([])
  powers2=np.array([])
  freqs=np.array([])

  deltaf = bw/nchan
  specs=getrshp()[:nchan]

  freqs1=np.array(range(int(num)*scan_range+1))*1./num*deltaf+nbin*deltaf-deltaf*2

  for freq in freqs1:
    print "setting freq ",str(freq)
    setfreq(freq)
    time.sleep(.5) 
    print runbash('./sg_ctrl freq')[0].split(' ')[0]

    spec_bm1=0
    spec_bin=0
    spec_bp1=0

    for k in range(10):
      spec=getrshp()[:1024]
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
'''

  for freq in freqs2:
    print "setting freq ",str(freq)
    setfreq(freq)
    time.sleep(2) 
    
:wq
owers2=np.append(powers2,getrshp()[:1024][100])

  for freq in freqs3:
    print "setting freq ",str(freq)
    setfreq(freq)
    time.sleep(2) 
    powers3=np.append(powers3,getrshp()[:1024][101])

'''

#  powers=append(powers1,powers2)
#  powers=append(powers,powers3) 

def getadc0():
  adc0=np.fromstring(fpga.snapshot_get('adcsnap0',man_trig=True,man_valid=True)['data'],dtype='<i1')
  return adc0

def getadc1():
  adc1=np.fromstring(fpga.snapshot_get('adcsnap0',man_trig=True,man_valid=True)['data'],dtype='<i1')
  return adc1

def getvacc():
  a=np.fromstring(fpga.snapshot_get('vaccout')['data'],dtype='>i4')[::4]
  return a[:32]

def getrshp():
  # grabs a snapshot of XX values
  # plot(getrshp()[:1024]
  rshp=np.fromstring(fpga.snapshot_get('rshpout')['data'],dtype='>i4')[::4]
  return rshp

def intrlv(ar1,ar2):
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

