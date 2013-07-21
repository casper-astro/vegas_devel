#!/opt/vegas/bin/python2.7

# these tests were taylored to the setup at the BWRC
import corr,time,struct
from subprocess import Popen, PIPE, STDOUT

roach = '192.168.40.80'
fpga=corr.katcp_wrapper.FpgaClient(roach,7147)
time.sleep(1)

def getadc0():
  adc0=np.fromstring(fpga.snapshot_get('adcsnap0',man_trig=True)['data'],dtype='<i1')
  return adc0

def getadc1():
  adc1=np.fromstring(fpga.snapshot_get('adcsnap0',man_trig=True)['data'],dtype='<i1')
  return adc1

def getstokes():
  stokes=np.fromstring(fpga.snapshot_get('stokesout',man_valid=True,man_trig=True)['data'],dtype='<i4')
  return stokes

def getvacc():
  a=np.fromstring(fpga.snapshot_get('vaccout')['data'],dtype='>i4')
  return a

def getvacc2():
  a=np.fromstring(fpga.snapshot_get('vaccout')['data'],dtype='>i4')[::4]
  return a[:32]

def getrshp():
  rshp=np.fromstring(fpga.snapshot_get('rshpout')['data'],dtype='>i4')[::4]
  return rshp

def getrshp1():
  rshp=np.fromstring(fpga.snapshot_get('rshpout1')['data'],dtype='>i4')[::4]
  return rshp

def rms(a):
  b=0
  for i in a: b=b+i**2 
  return np.sqrt(b/len(a))

def plothist(a):
  histo=hist(a,bins=100)
  return histo

def powers(a):
  powers=np.array([])
  for i in a:
    powers=append(powers,np.sqrt(i*i))
  return powers

def totalpower(a):
  b=0
  for i in a: b=b+i**2
  return b

def runbash(shell_command):
  event = Popen(shell_command, shell=True, stdin=PIPE, stdout=PIPE, stderr=STDOUT)
  return event.communicate()

def setfreq(freq):
  a=runbash('sg_ctrl freq '+str(freq))  
  time.sleep(.5)
  b=runbash('sg_ctrl freq')
  time.sleep(.5)
  return b[0].split(' ')[0]

def setampl(ampl):
  a=runbash('sg_ctrl ampl '+str(ampl))  
  time.sleep(.5)
  b=runbash('sg_ctrl ampl')
  time.sleep(.5)
  return b[0].split(' ')[0]

def setfftshift(val):
  fpga.write_int('fftshift',val)
  return(fpga.read_uint('fftshift'))

