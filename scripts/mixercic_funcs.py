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

def fill_mixer_bram(fpgaclient, n_inputs, cnt_r_name, mixer_name, limit, bramformat, data):
    fpgaclient.write_int(cnt_r_name, limit - 2 )  # This -2 because of the Rational a = b delay one clock for output and counter begin from 0
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
    for i in range(2,2**bramlength+1):
        for j in range(1,i/2+1):
            diff = abs(lof_input-Fs*j/(i*1.))
            if diff < lof_diff:
                lof_diff = diff
                lof_diff_m = i
                lof_diff_n = j
                print lof_diff,lof_diff_n,lof_diff_m,Fs*j/(i*1.)
                if diff == 0:
                    break
    return Fs*lof_diff_n/(lof_diff_m*1.), lof_diff_n, lof_diff_m



def lo_setup(fpgaclient, lo_f, bandwidth, n_inputs, cnt_r_name, mixer_name, bramlength):
    """
	lo_f: LO frequency
	bandwidth: ADC working bandwidth
	n_inputs: 2^n simultaneous inputs
	cnt_r_name: the name of the software register to control the upper limit of the address of the mixer bram
	bramlength: the depth of the brams in mixer
    """
    lof_output,lof_diff_n,lof_diff_num = calc_lof(bandwidth,bramlength,lof_input, 0, 0)
    if lo_f == 0:
	lo_input = constant_wave_gen(2**(bramlength+n_inputs))
    else:
	lo_input,tmp_a , tmp_b = wave_gen(lof_output, bandwidth*2, 2**(bramlength+n_inputs))
    bramforamt = '>'+str(lof_diff_num)+'I'
    print size(lo_input)
    fill_mixer_bram(fpgaclient, n_inputs, cnt_r_name, mixer_name, lof_diff_num, bramformat, lo_input)


def read_snaps(snap_depth = 13, adc_snap_depth = 11, mixer_snap_depth = 11):
    """
	TO DO: Clean up this function
    """
    #fpga.write_int('s32_s1p1re_ctrl', 0)
    fpga.write_int('s32_cicp1re_ctrl', 0)
    fpga.write_int('s64_adcp1_ctrl', 0)
    fpga.write_int('s1_mixer_snap64_ctrl', 0)
    time.sleep(1)

    #fpga.write_int('s32_s1p1re_ctrl', 7)
    fpga.write_int('s32_cicp1re_ctrl', 7)
    fpga.write_int('s64_adcp1_ctrl', 7)
    fpga.write_int('s1_mixer_snap64_ctrl', 7)
    time.sleep(1)

    #pol1_re = struct.unpack('>'+str(2**snap_depth)+'l',fpga.read('s32_s1p1re_bram', (2**snap_depth)*4))
    pol1_re = struct.unpack('>'+str(2**snap_depth)+'l',fpga.read('s32_cicp1re_bram', (2**snap_depth)*4))
    adc_data_m = struct.unpack('>'+str(2**adc_snap_depth*4)+'B',fpga.read('s64_adcp1_bram_msb', (2**adc_snap_depth)*4))
    adc_data_l = struct.unpack('>'+str(2**adc_snap_depth*4)+'B',fpga.read('s64_adcp1_bram_lsb', (2**adc_snap_depth)*4))
    mixer_data_m = struct.unpack('>'+str(2**mixer_snap_depth*4)+'b',fpga.read('s1_mixer_snap64_bram_msb', (2**mixer_snap_depth)*4))
    mixer_data_l = struct.unpack('>'+str(2**mixer_snap_depth*4)+'b',fpga.read('s1_mixer_snap64_bram_lsb', (2**mixer_snap_depth)*4))

    adc_data = arange(0, size(adc_data_m)*2, 1)
    adc_data[0::8] = adc_data_m[0::4]
    adc_data[1::8] = adc_data_m[1::4]
    adc_data[2::8] = adc_data_m[2::4]
    adc_data[3::8] = adc_data_m[3::4]
    adc_data[4::8] = adc_data_l[0::4]
    adc_data[5::8] = adc_data_l[1::4]
    adc_data[6::8] = adc_data_l[2::4]
    adc_data[7::8] = adc_data_l[3::4]
    mixer_data = arange(0, size(mixer_data_m)*2, 1)
    mixer_data[0::8] = mixer_data_m[0::4]
    mixer_data[1::8] = mixer_data_m[1::4]
    mixer_data[2::8] = mixer_data_m[2::4]
    mixer_data[3::8] = mixer_data_m[3::4]
    mixer_data[4::8] = mixer_data_l[0::4]
    mixer_data[5::8] = mixer_data_l[1::4]
    mixer_data[6::8] = mixer_data_l[2::4]
    mixer_data[7::8] = mixer_data_l[3::4]

    #pol1_re_np = array(pol1_re)
    #pol1_re = pol1_re_np.astype(int8)
    #pol1 = pol1_re[0::16]
    #print pol1_re[100:200:16]
    #print pol1_re[100:150]

    return pol1_re, adc_data, mixer_data



def test_plot(bandwidth, snap_deapth, dec_rate):
    """
	bandwidth: ADC working badnwidth
	snap_depth: depth of the snap blocks
	dec_rate: decimation rate
    """
    pol1_re, adc_data, mixer_data = read_snaps(snap_depth)
    pol1_re = pol1_re[0::(dec_rate/8)]
    x = []
    a = -bandwidth*1.0/dec_rate
    for i in range(size(pol1_re)):
        x.append(a)
        a = a + 2.0*(bandwidth/dec_rate)/size(pol1_re)
    pylab.ion()
    pylab.figure()

    pylab.subplot(221)
    pylab.title('ADC data')
    pylab.xlabel('N')
    pylab.plot(adc_data[100:500], '-o')
    pylab.hold(False)
	
    pylab.subplot(222)
    pylab.title('mixer_data')
    pylab.plot(mixer_data[100:200], '-o')
    pylab.hold(False)
        
    pylab.subplot(223)
    pylab.title('cic data')
    pylab.plot(pol1_re[100:200], '-o')
    pylab.hold(False)

    pol1_re_fft = fft(pol1_re)
    pylab.subplot(224)
    pylab.title('Mixed FFT,  Signal: '+str(signal_input)+'MHz,  LO: '+str(lof_output)+'MHz')
    pylab.xlabel('Frequency: MHz')
    pylab.semilogy(x,abs(pylab.fftshift((pol1_re_fft))))

    return pol1_re, x
