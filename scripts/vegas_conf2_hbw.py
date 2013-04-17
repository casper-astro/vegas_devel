#! /opt/vegas/bin/python2.7

import corr,time,struct 
roach = '192.168.40.80'

## IPs at Greenbank
#dest_ip = 192*(2**24)+168*(2**16)+3*(2**8)+15
#src_ip = 192*(2**24)+168*(2**16)+3*(2**8)+17

## IPs at BWRC
dest_ip = 10*(2**24)+145
src_ip = 10*(2**24)+4

dest_port = 60000

mac_base = (2 << 40) + (2<<32)
fabric_port = 60000

#acc_len=1023
acc_len=383
#acc_len=2046

fpga=corr.katcp_wrapper.FpgaClient(roach,7147)
time.sleep(1)

# ROACH@ 16 input ASIAA (DEBUG)
#boffile='v01_aa_rii_16r4t11f_ver114_2012_Sep_11_1555.bof'
#boffile='v02_aa_rii_16r4t12f_ver102_2012_Nov_15_1542.bof'

# ROACH@ 16 input SFP+
#boffile='v01_16r4t11f_ver137_2013_Jan_13_1830.bof'
#boffile='v01_16r4t11f_ver139_2013_Jan_13_1848.bof'
#boffile='v02_16r4t11f_ver103_2013_Jan_22_1839.bof'
#boffile='v01_16r4t11f_ver141_2013_Feb_19_1801.bof' # 1 subband - seems to work
#boffile='v13_16r128dr_ver111_2013_Mar_06_1933.bof'
#boffile='v02_16r4t11f_ver103_2013_Jan_22_1839.bof'
#boffile='v02_16r4t11f_ver104b_2013_Mar_18_1104.bof'
#boffile='v02_16r4t11f_ver105_2013_Mar_21_0316.bof'
#boffile='v02_16r4t11f_ver106_2013_Mar_22_0620.bof'
boffile='v02_16r4t11f_ver107_2013_Mar_22_1537.bof'

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
fpga.write_int('fftshift', 0b10111011111)

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

def getadc0():
  adc0=np.fromstring(fpga.snapshot_get('adcsnap0',man_trig=True)['data'],dtype='<i1')
  return adc0

def getadc1():
  adc1=np.fromstring(fpga.snapshot_get('adcsnap0',man_trig=True)['data'],dtype='<i1')
  return adc1

def getvacc():
  a=np.fromstring(fpga.snapshot_get('vaccout')['data'],dtype='>i4')[::4]
  return a[:32]

def getrshp():
  rshp=np.fromstring(fpga.snapshot_get('rshpout')['data'],dtype='>i4')[::4]
  return rshp

def getrshp1():
  rshp=np.fromstring(fpga.snapshot_get('rshpout1')['data'],dtype='>i4')[::4]
  return rshp

def getvaccout():
  vaccout1=np.fromstring(fpga.snapshot_get('vaccout1')['data'],dtype='>i4')
  vaccout2=np.fromstring(fpga.snapshot_get('vaccout2')['data'],dtype='>i4')

  a3=vaccout1[3::4]
  a2=vaccout1[2::4]
  a1=vaccout1[1::4]
  a0=vaccout1[::4]

  a7=vaccout2[3::4]
  a6=vaccout2[2::4]
  a5=vaccout2[1::4]
  a4=vaccout2[::4]

  vacc=np.array([]) 

  for i in range(len(a0)):

    vacc=np.append(vacc,a7[i]) 
    vacc=np.append(vacc,a6[i]) 
    vacc=np.append(vacc,a5[i]) 
    vacc=np.append(vacc,a4[i]) 
    vacc=np.append(vacc,a3[i]) 
    vacc=np.append(vacc,a2[i]) 
    vacc=np.append(vacc,a1[i]) 
    vacc=np.append(vacc,a0[i]) 

  return vacc

def intrlv(ar1,ar2):
  ar3 = np.zeros((len(ar1) + len(ar2)))
  ar3[0::2] = ar1
  ar3[1::2] = ar2
  return ar3 
       
#def getrshp2():
#  fpga.write_int('rshpout_ctrl',0)
#  fpga.write_int('rshpout_ctrl',1)
#  fpga.write_int('rshpout_ctrl',0)
#  a=fpga.read('rshpout_bram',16384)
#  b=np.fromstring(a,dtype='uint32')
#  b1=struct.unpack('>4096I',a)
#  return b1[::4]

def plotrshp():
  fpga.write_int('rshpout_ctrl',0)
  fpga.write_int('rshpout_ctrl',1)
  fpga.write_int('rshpout_ctrl',0)
  a=fpga.read('rshpout_bram',16384)
  b=np.fromstring(a,dtype='uint32')
  b1=struct.unpack('>4096I',a)
  b1=b1[::4]
  plot(b1)
  show() 

def getstokes():
  stokes=np.fromstring(fpga.snapshot_get('stokesout',man_valid=True,man_trig=True)['data'],dtype='>i4')[::4]
  return stokes

def extract(reg,offset,bitwidth):
  reg = reg >> offset
  mask=1 
  for i in range(bitwidth-1):
    mask=(mask << 1) + 1
  return reg & mask

def get_debug():
  #debug_bram=np.fromstring(fpga.snapshot_get('debug',man_valid=True,man_trig=True)['data'],dtype='>i4')
  debug_bram=np.fromstring(fpga.snapshot_get('debug')['data'],dtype='>i4')
  return debug_bram

def get_valid(offset):
  #debug_bram=get_debug()
  bitwidth=1
  validarr=np.array([])
  for i in range(len(debug_bram)):
    val=extract(debug_bram[i],offset,bitwidth)
    validarr=np.append(validarr,val)
  return validarr


def makedebugdict(debug_bram):

  debugdict = {'data_valid':[],
	       'eoh':[],
	       'eof':[],
	       'spead_st':[],
	       'data_valid_spead':[],
	       'reshape_state':[],
	       'data_enable':[],
	       'spead_rst':[],
	       'master_rst':[],
	       'tge_rst':[],
	       'valid':[],
	       'txack':[]} 

  for i in range(size(debug_bram)):
    debugdict['data_valid'].append(extract(debug_bram[i],15,1))
    debugdict['eoh'].append(extract(debug_bram[i],14,1))
    debugdict['eof'].append(extract(debug_bram[i],13,1))
    debugdict['spead_st'].append(extract(debug_bram[i],10,3))
    debugdict['data_valid_spead'].append(extract(debug_bram[i],9,3))
    debugdict['reshape_state'].append(extract(debug_bram[i],6,3))
    debugdict['data_enable'].append(extract(debug_bram[i],5,1))
    debugdict['spead_rst'].append(extract(debug_bram[i],4,1))
    debugdict['master_rst'].append(extract(debug_bram[i],3,1))
    debugdict['tge_rst'].append(extract(debug_bram[i],2,1))
    debugdict['valid'].append(extract(debug_bram[i],1,1))
    debugdict['txack'].append(extract(debug_bram[i],0,1))

  return debugdict

def plotdebug(debubdict):
  
  f, xarr = subplots(6,sharex=True)
  dd=debugdict
  xarr[0].plot(dd['data_valid'])
  xarr[0].set_title('data_valid')

  xarr[1].plot(dd['eoh'])
  xarr[1].set_title('eoh')

  xarr[2].plot(dd['eof'])
  xarr[2].set_title('eof')

  xarr[3].plot(dd['spead_st'])
  xarr[3].set_title('spead_st')

  xarr[4].plot(dd['data_valid_spead'])
  xarr[4].set_title('data_valid_spead')

  xarr[5].plot(dd['reshape_state'])
  xarr[5].set_title('reshape_state')

reset()
#debug_bram = getdebug()


#np.fromstring(u.snapshot_get('vaccout',man_valid=True,man_trig=True)['data'],dtype='>i4')

#np.fromstring(u.snapshot_get('rshpout')['data'],dtype='>i4')[::2]


