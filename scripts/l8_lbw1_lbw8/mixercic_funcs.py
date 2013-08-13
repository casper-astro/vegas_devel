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

def get_mode():
    """ get the current mode setting in register mode_sel, assign the value to global var mode_sel and return it """
    global mode_sel
    mode_sel = fpga.read_int('mode_sel')
    return mode_sel

def set_mode(mode_sel):
    """ set mode to mode_sel (0 for l8_lbw8, 1 for l8_lbw1) """
    fpga.write_int('mode_sel', mode_sel)

def setgain(subband, gain1, gain2):
    '''
     assign gain1 & gain2 for the two pols in given subband
    '''
    fpga.write_int('s'+str(subband)+'_quant_gain1',gain1)
    fpga.write_int('s'+str(subband)+'_quant_gain2',gain2)

def get_mixer_lmt(mixer_name):
    lmt = fpga.read_int(mixer_name+'_mixer_cnt')
    return lmt

def gbeconf():
    fpga.tap_start('tap0','gbe0',mac_base+src_ip,src_ip,fabric_port)

def wave_gen(lo_f, sample_f, size):
    """
    Parameters:
        lo_f: float; in Hz (or in whatever unit as long as it's the same as sample_f)
        sample_f: float; in Hz (or just keep consistatn with lo_f)
        size: length of brams to fill (all the brams in multi-paralle input cases)
    Return values:
        resutls: an int array of 32-bit integers.
        siniw: an int array of 16-bit integers
        cosiw: an int array of 16-bit integers
    Notes:
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
        sincos = int32(sini)*(2**16) + cosi
        #"""
        siniw.append(sini)
        cosiw.append(cosi)
        sinfw.append(sinf)
        cosfw.append(cosf)
        #"""
        result.append(sincos)
    return result, siniw, cosiw


def constant_wave_gen(size):
    """ Generate a constant array of given size """
    result = [0x7fff7fff] * size
    return result

def fill_mixer_bram(mixer_name, limit, bramformat, data):
    '''
    Parameters:
	mixer_name: a string denoting the name of the subband
	limit: a number, setting the limit of the portions of brams to be filled
        bramformat: the format of data to be packed into brams
        data: the actual data to be filled into brams
    Return values: 
        NONE
    Global variables used:
	fpga
	n_inputs
	t_sleep
    Notes: 
        limit == len(data)/(2**n_inputs)	
    '''
    fpga.write_int(mixer_name+'_mixer_cnt', limit - 2 )  # This -2 because of the Rational a = b delay one clock for output and counter begin from 0
    time.sleep(t_sleep)
    for i in range(2**n_inputs):
	fpga.write(mixer_name+'_lo_'+str(i)+'_lo_ram', struct.pack(bramformat, *data[i::(2**n_inputs)]))
	time.sleep(t_sleep)
	print('done with '+mixer_name+'_'+str(i))

def calc_lof(lof_input):
    """
    Function:
        Calculate the closest frequency of lof_input, return the exact
        working frequency of the mixer, the numertor and 
        denominator of the combination.
        The available frequency combination can be: 
        (1/2,1/3,1/4,1/5,2/5,1/6/,1/7,2/7,3/7,...,1/2^bramlength,2/2^bramlength,...) times Fs
    Global variables used:
        bw: ADC sampling bandwidth, half of ADC sampling frequency, e.g. 600.0(MHz)
        bramlength: Mixer BRAM length: 2^bramlength, e.g. 10
	n_inputs: number of parallel inputs: 2^n_inputs, e.g. 8
    Parametrs:
        lof_input: The demanded mixer frequency, e.g. 241.0(MHz)
    Returns:
        lof_ctual: Mixer working frequency
        lof_diff_n: Numerator ( 2*bw*N/D )
        lof_diff_m: Denominator ( 2*bw*N/D)
    Notes:
      How do we find the numberator and denominator? e.g. say we have 8 bram of length 2**10 (for 8 parallel inputs), we can choose to
      use a fraction of them (this upperboudn of address for all subbands combined would be lof_diff_m, the denominator, which HAS
      to  be a multiple of 8, the number of parralel inputs) to store lof_diff_n cycles of sine/cosine
      waves, which has frequency of (lof_diff_n*Fs/lof_diff_m)
    """
    Fs = 2*bw
    lof_diff = 1000
    lof_diff_m = 0
    lof_diff_n = 0
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

def lo_setup(lo_f, mixer_name):
    """
    Parameters:
	lo_f: LO frequency (desired value, might not be able to achieve it precisely)
        mixer_name: the name of the software register to control the upper limit of the address of the mixer bram
    Global variables used:
	fpga: handle
	bw: ADC working bandwidth
	n_inputs: 2^n simultaneous inputs
	bramlength: the depth of the brams in mixer
    Return:
	lof_output: tells user what the actual mixing frequency (LO) is
	lo_wave: the generated sine wave to be filled into the brams
    """
    print 'Setting up subband...'+mixer_name
    if lo_f == 0:
	lof_diff_num = 2**(bramlength+n_inputs)
	lo_wave = constant_wave_gen(2**(bramlength+n_inputs))
	lof_output = 0
    else:
	lof_output,lof_diff_n,lof_diff_num = calc_lof(lo_f)
	lo_wave, tmp_a, tmp_b = wave_gen(lof_output, bw*2, lof_diff_num)
    bramformat = '>'+str(lof_diff_num/(2**n_inputs))+'I'
    #print size(lo_wave), ' bramformat', bramformat
    fill_mixer_bram(mixer_name, lof_diff_num/(2**n_inputs), bramformat, lo_wave)
    return lof_output, lo_wave

def lo_adjust(i, new_lo):
  """ set Nth subband to LO """
  lo_actual, lo_wave = lo_setup(new_lo, 's'+str(i))
  return lo_actual, lo_wave


