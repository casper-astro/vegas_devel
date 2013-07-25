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
    '''
    Parameters:
       data: an array of data to plot
       lo_f: LO
       bw: bandwidth
       dec_rate: decimation rate (not neccessarily the same as the total decimation rate in the design
    Return:
       NONE
    '''
    f_index = np.linspace(lo_f - bw/(1.*dec_rate), lo_f + bw/(1.*dec_rate), len(
data))
    plot(f_index, 10*np.log10(np.abs(np.fft.fftshift(np.fft.fft(data)))))

def plotfft(ax, data, lo_f, bw, dec_rate):
    '''
    Parameters:
       ax: an axis of figure to plot
       data: an array of data to plot
       lo_f: LO
       bw: bandwidth
       dec_rate: decimation rate (not neccessarily the same as the total decimation rate in the design
    Return:
       NONE
    '''
    f_index = np.linspace(lo_f - bw/(1.*dec_rate), lo_f + bw/(1.*dec_rate), len(
data))
    f_index = f_index - bw/(1.*dec_rate*len(data))
    ax.plot(f_index, 10*np.log10(np.abs(np.fft.fftshift(np.fft.fft(data)))))

def makecomplex(data0, data1):
    """ Given two real 1-D arrays and return a complex array """
    data_len = len(data0)
    complexdata = np.zeros(data_len, dtype=np.complex64)
    complexdata.real = data0
    complexdata.imag = data1
    return complexdata

def get_mixer_ri(pol, reim):
    """ Read the output of mixer, given: (1) pol ; (2) re/im ***subband1 """
    mixer = np.fromstring(fpga.snapshot_get('s1_mixer_p'+str(pol)+reim,man_trig=True, man_valid=True)['data'],dtype='>i1')
    return mixer

def get_mixer(pol):
    '''
    Get the output of mixer @subband 1
    Parameters:
	pol: 1 or 2
    Return values:
	mixer_re: real
	mixer_im: imag
    Notes: as of ver121_01, the re/im streams of mixers are not in sync
    '''
    mixer_re = get_mixer_ri(pol, 're')
    mixer_im = get_mixer_ri(pol, 'im')
    return mixer_re, mixer_im

def getadc(i):
    """ Read the output of adc, given (i) *** i=0 OR i=1  """
    adc=np.fromstring(fpga.snapshot_get('adcsnap'+str(i),man_trig=True,man_valid=True)['data'],dtype='<i1')
    return adc

def get_dr16(pol):
    '''
    Read the output of first CIC stage (dec_rate=16) @ subband 1
    Parameters: pol (1 or 2)
    Return: dr16_out: a complex array
    '''
    dr16=np.fromstring(fpga.snapshot_get('s1_p'+str(pol)+'_dr16', man_trig=True,
 man_valid=True)['data'], dtype='>i8')
    dr16_re = dr16[0::2]
    dr16_im = dr16[1::2]
    dr16_out = makecomplex(dr16_re, dr16_im)
    return dr16_out

def get_1stcic(pol):
    '''
    Read the output of the first CIC (before Halfband, dec_rate=32) @subband1
    Parameters: pol (1 or 2)
    Returns: 
	first_cic_out (complex array, 32-bit re/im)
	first_cic_full_out (complex array,64-bit re/im)
    '''
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
    '''
    Read the output of the halfband filter (so far, dec_rate=32) @subband1
    Parameters: pol1 (1 or 2)
    Returns: a complex array
    '''
    halfband = np.fromstring(fpga.snapshot_get('s1_p'+str(pol)+'_halfband', man_trig=True, man_valid=True)['data'], dtype='>i4')
    hb_re = halfband[0::4]
    hb_im = halfband[1::4]
    hb_out = makecomplex(hb_re, hb_im)
    return hb_out

def get_subband1(pol):
  ''' 
  This is for the output of the first subband
  Parameters:
     pol: 1 or 2
  Return values:
     s1p: complex array (8-bit re/im)
  Notes:
     Compare to get_chc(pol), this one has only 8-bit while the other has 32 bits
  '''
  s1p_raw = np.fromstring(fpga.snapshot_get('s1p'+str(pol)+'_snap',man_trig=True,man_valid=True)['data'],dtype='<i1')
  s1p_re = s1p_raw[0::2*(dec_rate/(2**n_inputs))]
  s1p_im = s1p_raw[1::2*(dec_rate/(2**n_inputs))]
  s1p = makecomplex(s1p_re, s1p_im)
  return s1p

def get_chc(pol):
  ''' 
  This is for the output of the first subband
  Parameters: pol1 (1 or 2)
  Returns: a complex array
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
    '''
    Plot the output of the first CIC stage (dec_rate = 16)
    Two pols, complex spectra
    '''
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
    '''
    Plot the output of the first CIC (dec_rate = 32)
    Two pols, complex spectra
    '''
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
    '''
    Plot the output of the halfband filter (as of now, dec_rate=32 due to the CIC filter before the halfband filter)
    Two pools, complex spectra
    '''
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
    '''
    Plot the output of the combined filter (dec_rate = 128)
    '''
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

def filterresponse(pol, lo_f, scan_range=1, skip=50):
  '''
  Plot the filter response of the CIC-Halfband-CIC filter (subband1)
  Global variables:
    brd_clk: ADC bandwidth
    dec_rate: the decimation rate of the 
  Parameters:
    pol: which pol to plot
    lo_f: LO
    scan_range: how many times of bandpass range to scan
    skip: how to select test frequencies - skip how many points
  Return values:
    resp: response array
    freqs: an array of frequencies corresponding to the resp array
    specs: a 2-D array containing corresponding spectra 
  '''
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
    setampl(-15) #BWRC
    setfreq(freq) #BWRC
    time.sleep(.3)
    print runbash('./sg_ctrl freq')[0].split(' ')[0]  #BWRC
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


