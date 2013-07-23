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
	#print('done with '+str(i))



def calc_lof(Fs,bramlength,n_inputs,lof_input,lof_diff_n,lof_diff_m):
    """
    Function:
        Calculate the closest frequency of lof_input, return the exact
        working frequency of the mixer, the numertor and 
        denominator of the combination.
        The available frequency combination can be: 
        (1/2,1/3,1/4,1/5,2/5,1/6/,1/7,2/7,3/7,...,1/2^bramlength,2/2^bramlength,...) times Fs
    # How do we find the numberator and denominator? e.g. say we have 8 bram of length 2**10 (for 8 parallel inputs), we can choose to
    #    use a fraction of them (this upperboudn of address for all subbands combined would be lof_diff_m, the denominator) to store lof_diff_n cycles of sine/cosine
    #    waves, which has frequency of (lof_diff_n*Fs/lof_diff_m)
    Parameters:
        Fs: ADC sampling frequency, e.g. 600.0(MHz)
        bramlength: Mixer BRAM length: 2^bramlength, e.g. 10
	n_inputs: number of parallel inputs: 2^n_inputs, e.g. 8
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
	lof_output,lof_diff_n,lof_diff_num = calc_lof(bandwidth*2,bramlength,n_inputs,lo_f, 0, 0)
	lo_wave, tmp_a, tmp_b = wave_gen(lof_output, bandwidth*2, lof_diff_num)
    bramformat = '>'+str(lof_diff_num/(2**n_inputs))+'I'
    #print size(lo_wave), ' bramformat', bramformat
    fill_mixer_bram(fpgaclient, n_inputs, mixer_name, lof_diff_num/(2**n_inputs), bramformat, lo_wave)
    return lof_output


