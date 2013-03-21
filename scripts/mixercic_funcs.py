import numpy, corr, time, struct, sys, logging, pylab, matplotlib, scipy
from numpy import *
from scipy import *

def exit_fail():
    print 'FAILED. Log entries:\n',lh.printMessages()
    try:
        fpga.stop()
    except: pass
    raise
    exit()

def exit_clean():
    try:
        fpga.stop()
    except: pass
    exit()

def set_gain_sg(fpgaclient, gain_r_name, gain):
    fpgaclient.write(gain_r_name, gain)

def wave_gen(lo_f, sample_f, size):
    """
	Assuming the data type is FIX_16_15 in BRAM
    """
    sinfw = []
    cosfw = []
    siniw = []
    cosiw = []
    result = []
    for i in range(size):
        sinf = -sin(2*pi*lo_f/(sample_f*1.)*i)
        sini = int16(sinf*32767)
        cosf = cos(2*pi*lo_f/(sample_f*1.)*i)
        cosi = int16(cosf*32767)
        sincos = int32(sini)*2**16 + cosi
        #"""
        siniw.append(sini)
        cosiw.append(cosi)
        sinfw.append(sinf)
        cosfw.append(cosf)
        #"""
        result.append(sincos)
    return result, siniw, cosiw


def constant_wave_gen(size):
    result = [0x7fff7fff] * size
    return result

def fill_mixer_bram(fpgaclient, n_inputs, mixer_name, limit, bramformat, data):
    fpgaclient.write_int(mixer_name+'_mixer_cnt', limit - 2 )  # This -2 because of the Rational a = b delay one clock for output and counter begin from 0
    for i in range(2**n_inputs):
	fpgaclient.write(mixer_name+'_lo_'+str(i)+'_lo_ram', struct.pack(bramformat, *data[i::(2**n_inputs)]))
	print('done with '+str(i))



def calc_lof(Fs,bramlength,lof_input,lof_diff_n,lof_diff_m):
    """
    Function:
        Calculate the closest frequency of lof_input, return the exact
        working frequency of the mixer, the numertor and 
        denominator of the combination.
        The available frequency combination can be: 
        (1/2,1/3,1/4,1/5,2/5,1/6/,1/7,2/7,3/7,...,1/2^bramlength,2/2^bramlength,...) times Fs
    # How do we find the numberator and denominator? e.g. say we have a bram of length 2**10, we can choose to
    #    use a fraction of it (this would be lof_diff_m, the denominator) to store lof_diff_n cycles of sine/cosine
    #    waves, which has frequency of (lof_diff_n*Fs/lof_diff_m)
    Parameters:
        Fs: ADC sampling frequency, e.g. 600.0(MHz)
        bramlength: Mixer BRAM length: 2^bramlength, e.g. 10
        lof_input: The demanded mixer frequency, e.g. 241.0(MHz)
        lof_diff_n: The closest frequency numerator
        lof_diff_m: The closest frequency denominator
    Returns:
        Mixer working frequency
        Numerator
        Denominator
    """
    lof_diff = 1000
    for i in range(2**n_inputs,2**(bramlength+n_inputs)+1, 2**n_inputs):
        for j in range(1,i/2+1):
            diff = abs(lof_input-Fs*j/(i*1.))
            if diff < lof_diff:
                lof_diff = diff
                lof_diff_m = i
                lof_diff_n = j
                #print lof_diff,lof_diff_n,lof_diff_m,Fs*j/(i*1.)
                if diff == 0:
                    break
    lof_actual = Fs*lof_diff_n/(lof_diff_m*1.)
    print 'Mixing frequency (user input): '+str(lof_input)+'MHz'
    print 'Mixing frequency (actual achivable):'+str(lof_actual)+'MHz'
    return lof_actual, lof_diff_n, lof_diff_m

def lo_setup(fpgaclient, lo_f, bandwidth, n_inputs, mixer_name, bramlength):
    """
	lo_f: LO frequency (desired value, might not be able to achieve it precisely)
	bandwidth: ADC working bandwidth
	n_inputs: 2^n simultaneous inputs
	cnt_r_name: the name of the software register to control the upper limit of the address of the mixer bram
	bramlength: the depth of the brams in mixer
    Return:
	lof_output: tells user what the actual mixing frequency (LO) is
    """
    print 'Setting up subband...'+mixer_name
    if lo_f == 0:
	lof_diff_num = 2**(bramlength+n_inputs)
	lo_wave = constant_wave_gen(2**(bramlength+n_inputs))
	lof_output = 0
    else:
	lof_output,lof_diff_n,lof_diff_num = calc_lof(bandwidth,bramlength,lo_f, 0, 0)
	lo_wave, tmp_a, tmp_b = wave_gen(lof_output, bandwidth*2, lof_diff_num)
    bramformat = '>'+str(lof_diff_num/(2**n_inputs))+'I'
    #print size(lo_wave), ' bramformat', bramformat
    fill_mixer_bram(fpgaclient, n_inputs, mixer_name, lof_diff_num/(2**n_inputs), bramformat, lo_wave)
    return lof_output


def read_snaps(fpgaclient, bram_length = 12):
    """
	As of Feb 18, works for v13_16r128dr_ver104.mdl
	*** First four (cic*_*_snap) have 1/2 referring to 2 pols
	cic1_re_snap: 32 bit data, depth = 12
	cic1_im_snap: ^
	cic2_re_snap: ^
	cic2_im_snap: ^
	adc1_snap: 8 bit data -> 128 bit (16 parallel inputs), depth = 12
	adc2_snap: ^
	s1p1re_snap: 8 bit data, depth = 12
	s1_firstcic_snap: 32 bit data, depth = 12
	s1_halfband_snap: 32 bit data, depth = 12
	s1_mixer_snap: 8 bit -> 128 bit data, depth = 12
    """
    cic1_re = fromstring(fpgaclient.snapshot_get('cic1re_snap')['data'],dtype='int32')
    cic1_im = fromstring(fpgaclient.snapshot_get('cic1im_snap')['data'],dtype='int32')  
    cic2_re = fromstring(fpgaclient.snapshot_get('cic2re_snap')['data'],dtype='int32')
    cic2_im = fromstring(fpgaclient.snapshot_get('cic2im_snap')['data'],dtype='int32')
    adc1 = fromstring(fpgaclient.snapshot_get('adc1_snap')['data'], dtype='int8')
    adc2 = fromstring(fpgaclient.snapshot_get('adc2_snap')['data'], dtype='int8')
    s1p1re = fromstring(fpgaclient.snapshot_get('s1p1re_snap')['data'], dtype='int8')
    s1_firstcic = fromstring(fpgaclient.snapshot_get('s1_firstcic_snap')['data'], dtype='int32')
    s1_halfband = fromstring(fpgaclient.snapshot_get('s1_halfband_snap')['data'], dtype='int32')
    s1_mixer = fromstring(fpgaclient.snapshot_get('s1_mixer_snap')['data'], dtype='int8')
    data_dict = {'cic1_re': cic1_re,
		 'cic1_im': cic1_im,
		 'cic2_re': cic2_re, 
		 'cic2_im': cic2_im,
		 'adc1': adc1,
		 'adc2': adc2,
		 's1p1re': s1p1re, 
		 's1_firstcic': s1_firstcic,
		 's1_halfband': s1_halfband,
		 's1_mixer': s1_mixer}
    return data_dict


def test_plot(fpgaclient, bandwidth, snap_depth, dec_rate, n_inputs, signal_input, lof):
    """
	fpgaclient: 
	bandwidth: ADC working badnwidth
	snap_depth: depth of the snap blocks
	dec_rate: decimation rate
	n_inputs: number of simutaneous inputs (2^n)
	signal_input: known frequency of the test tone
	lof: lo frequency
    """
    data_dict = read_snaps(fpgaclient, snap_depth)

    pol1_re = data_dict['cic1_re']
    pol1_re = pol1_re[0::(dec_rate/(2**n_inputs))]
    x = []
    a = -bandwidth*1.0/dec_rate
    for i in range(size(pol1_re)):
        x.append(a)
        a = a + 2.0*(bandwidth/dec_rate)/size(pol1_re)
    pylab.ion()
    pylab.figure()

    pylab.subplot(231)
    pylab.title('ADC data')
    pylab.xlabel('N')
    pylab.plot(data_dict['adc1'][100:500], '-o')
    pylab.hold(False)
	
    pylab.subplot(232)
    pylab.title('mixer_data')
    pylab.plot(data_dict['s1_mixer'][100:200], '-o')
    pylab.hold(False)
        
    pylab.subplot(233)
    pylab.title('cic data')
    pylab.plot(pol1_re[100:200], '-o')
    pylab.hold(False)

    pol1_re_fft = fft(pol1_re)
    pylab.subplot(234)
    pylab.title('Mixed FFT,  Signal: '+str(signal_input)+'MHz,  LO: '+str(lof)+'MHz')
    pylab.xlabel('Frequency: MHz')
    pylab.semilogy(x,abs(pylab.fftshift((pol1_re_fft))))

    pylab.subplot(235)
    pylab.title('first cic data')
    pylab.plot(data_dict['s1_firstcic'][100:200], '-o')
    pylab.hold(False)

    pylab.subplot(236)
    pylab.title('halfband data')
    pylab.plot(data_dict['s1_halfband'][100:200], '-o')
    return pol1_re, x
