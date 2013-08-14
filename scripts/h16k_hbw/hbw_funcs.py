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

  nbin = nbin-1 # temporal hack fix for ver107_01
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
  rshp=np.fromstring(fpga.snapshot_get('rshpout')['data'],dtype='>i4')[0::2]
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


