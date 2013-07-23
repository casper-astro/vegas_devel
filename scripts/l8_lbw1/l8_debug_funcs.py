def makecomplex(data0, data1):
    data_len = len(data0)
    complexdata = np.zeros(data_len, dtype=np.complex64)
    complexdata.real = data0
    complexdata.imag = data1
    return complexdata

def get_mixer(fpga, subband, pol, reim):
    mixer_out = np.fromstring(fpga.snapshot_get('s'+str(subband)+'_mixer_p'+str(pol)+reim)['data'],dtype='>i1')
    return mixer_out

def get_re_dr16(pol):
    '''
    subband 1, first CIC filter (dec_rate=16)
    '''
    p1re_dr16=np.fromstring(fpga.snapshot_get('s1_p'+str(pol)+'re_dr16', man_trig=True, man_v
alid=True)['data'], dtype=np.int64)
    return p1re_dr16

def get_im_dr16(pol):
    '''
    subband 1, first CIC filter (dec_rate=16)
    '''
    p1im_dr16=np.fromstring(fpga.snapshot_get('s1_p'+str(pol)+'im_dr16', man_trig=True, man_v
alid=True)['data'], dtype=np.int64)
    return p1im_dr16

def get_p_dr16(pol):
    re_data = getp1re_dr16(pol)
    im_data = getp1im_dr16(pol)
    dr16 = makecomplex(re_data, im_data)
    return dr16


def get_1stcic_ri(pol, reim):
    first_cic = np.fromstring(fpga.snapshot_get('s1_p'+str(pol)+reim+'_firstcic', man_trig=True, man_valid=True)['data'],dtype=np.int32)
    first_cic_full = np.fromstring(fpga.snapshot_get('s1_p'+str(pol)+reim+'_firstcic_full', man_trig=True, man_valid=True)['data'],dtype=np.int64)
    return first_cic, first_cic_full

def get_1stcic(pol):
    re_data, re_full = get_1stcic_ri(pol, 're')
    im_data, im_full = get_1stcic_im(pol, 'im')
    first_cic = makecomplex(re_data, im_data)
    first_cic_full = makecomplex(re_full, im_full)

'''
def get_dr16(fpga, subband, pol, reim):
    dr16 = np.fromstring(fpga.snapshot_get('s'+str(subband)+'_p'+str(pol)+reim+'_dr16', man_trig=True, man_valid=True)['data'], dtype='>i8')
    return dr16
'''

def get_halfband_ri(pol, reim):
    halfband = np.fromstring(fpga.snapshot_get('s1_p'+str(pol)+reim+'_halfband', man_trig=True, man_valid=True)['data'], dtype=np.int64)
    return halfband

def get_halfband(pol):
    re_data = get_halfband_ri(pol, 're')
    im_data = get_halfband_ri(pol, 'im')
    halfband = makecomplex(re_data, im_data)
    return halfband

def get_chc_ri(pol, reim):
    chc_raw = np.fromstring(fpga.snapshot_get('cic'+str(pol)+reim+'_snap', man_trig=True, man_valid=True)['data'], dtype=np.int32)
    chc = chc_raw[0::dec_rate/(2**n_inputs)]
    return chc

def get_chc(pol):
    re_data = get_chc_ri(pol, 're')
    im_data = get_chc_ri(pol, 'im')
    chc = makecomplex(re_data, im_data)
    return chc

def plotfft(ax, data, lo_f, bw, dec_rate):
    f_index = np.linspace(lo_f - bw/(1.*dec_rate), lo_f + bw/(1.*dec_rate), len(data))
    ax.plot(f_index, 10*np.log10(np.abs(np.fft.fftshift(np.fft.fft(data)))))

def plot_adc(fpga, bw):
    data0 = getadc0()
    data1 = getadc1()
    f, (ax1, ax2) = subplots(2)
    ax1.plot(data0[100:200],'-o')
    ax2.plot(data1[100:200],'-o')
    show()
    f, (ax1, ax2) = subplots(2)
    adcdata0 = makecomplex(data0, data0)
    adcdata1 = makecomplex(data1, data1)
    plotfft(ax1, adcdata0, 0, bw, 1)
    plotfft(ax2, adcdata1, 0, bw, 1)
    show()

def plot_mixer(fpga, subband, lo_f, bw):
    m_p1re = get_mixer(fpga, subband, 1, 're')
    m_p1im = get_mixer(fpga, subband, 1, 'im')
    m_p2re = get_mixer(fpga, subband, 2, 're')
    m_p2im = get_mixer(fpga, subband, 2, 'im')
    f, (ax1, ax2, ax3, ax4) = subplots(4)
    ax1.plot(m_p1re[0:100],'-o')
    ax2.plot(m_p1im[0:100],'-o')
    ax3.plot(m_p2re[0:100],'-o')
    ax4.plot(m_p2im[0:100],'-o')
    show()
    f, (ax1, ax2) = subplots(2)
    m_p1 = makecomplex(m_p1im, m_p1re)
    plotfft(ax1, m_p1, lo_f, bw, 1)
    m_p2 = makecomplex(m_p2re, m_p2im)
    plotfft(ax2, m_p2, lo_f, bw, 1)    
    show()

def plot_dr16(fpga, subband, lo_f, bw, dec_rate):
    dr16_p1_re = get_dr16(fpga, subband, 1, 're')
    dr16_p1_im = get_dr16(fpga, subband, 1, 'im')
    dr16_p2_re = get_dr16(fpga, subband, 2, 're')
    dr16_p2_im = get_dr16(fpga, subband, 2, 'im')
    f, (ax1, ax2) = subplots(2)
    dr16_p1 = makecomplex(dr16_p1_re, dr16_p1_im)
    plotfft(ax1, dr16_p1, lo_f, bw, dec_rate)
    dr16_p2 = makecomplex(dr16_p2_re, dr16_p2_im)
    plotfft(ax2, dr16_p2, lo_f, bw, dec_rate)

def plot_1stcic(fpga, subband, lo_f, bw, dec_rate):
    c_p1re, c_p1re_full = get_1stcic(fpga, subband, 1, 're')
    c_p1im, c_p1im_full = get_1stcic(fpga, subband, 1, 'im')
    c_p2re, c_p2re_full = get_1stcic(fpga, subband, 2, 're')
    c_p2im, c_p2im_full = get_1stcic(fpga, subband, 2, 'im')
    f, (ax1, ax2) = subplots(2)
    c_p1 = makecomplex(c_p1re, c_p1im)
    c_p2 = makecomplex(c_p2re, c_p2im)
    plotfft(ax1, c_p1, lo_f, bw, dec_rate)
    plotfft(ax2, c_p2, lo_f, bw, dec_rate)
    f, (ax1, ax2) = subplots(2)
    c_p1f = makecomplex(c_p1re_full, c_p1im_full)
    c_p2f = makecomplex(c_p2re_full, c_p2im_full)
    plotfft(ax1, c_p1f, lo_f, bw, dec_rate)
    plotfft(ax2, c_p2f, lo_f, bw, dec_rate)

def plot_hb(fpga, subband, lo_f, bw, dec_rate):
    hb_p1re = get_halfband(fpga, subband, 1, 're')
    hb_p1im = get_halfband(fpga, subband, 1, 'im')
    hb_p2re = get_halfband(fpga, subband, 2, 're')
    hb_p2im = get_halfband(fpga, subband, 2, 'im')
    hb_p1_len = len(hb_p1re)
    hb_p1 = np.zeros(hb_p1_len, dtype=np.complex64)
    hb_p1.real = hb_p1re.astype(np.float)
    hb_p1.imag = hb_p1im.astype(np.float)
    subplot(211)
    plotfft(hb_p1, lo_f, bw, dec_rate)
    hb_p2_len = len(hb_p2re)
    hb_p2 = np.zeros(hb_p2_len, dtype=np.complex64)
    hb_p2.real = hb_p2re.astype(np.float)
    hb_p2.imag = hb_p2im.astype(np.float)
    subplot(212)
    plotfft(hb_p2, lo_f, bw, dec_rate)


def plot_chc(fpga, subband, lo_f, bw, dec_rate, n_inputs):
    chc_p1re = get_chc(fpga, subband, 1, 're', dec_rate, n_inputs)
    chc_p1im = get_chc(fpga, subband, 1, 'im', dec_rate, n_inputs)
    chc_p2re = get_chc(fpga, subband, 2, 're', dec_rate, n_inputs)
    chc_p2im = get_chc(fpga, subband, 2, 'im', dec_rate, n_inputs)
    f, (ax1, ax2) = subplots(2)
    chc_p1 = makecomplex(chc_p1re, chc_p1im)
    chc_p2 = makecomplex(chc_p2re, chc_p2im)
    plotfft(ax1, chc_p1, lo_f, bw, dec_rate)
    plotfft(ax2, chc_p2, lo_f, bw, dec_rate)
    show()
    

