#!/usr/bin/env ipython


##################################################################
# Script Information
# This script can be used as iADC(2GSps) board testing
# Connect the input with a frequence such as 13MHz and clock
# as 800MHz
# 20120529@CASPER.Berkeley
##################################################################
import numpy, corr, time, struct, sys, logging, pylab, matplotlib, scipy
from numpy import *
from scipy import *




##################################################################





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

def wave_gen(lo_f, sample_f, size):
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
        """
        siniw.append(sini)
        cosiw.append(cosi)
        sinfw.append(sinf)
        cosfw.append(cosf)
        """
        result.append(sincos)
    """
    """
    return result


def constant_wave_gen(size):

    result = [0x7fff7fff] * size
    return result


def set_gain_sg(fpgaclient, gain):
    fpgaclient.write('subband1_quant_gain', gain)


def fill_mixer_bram(fpgaclient, subbandname, bramformat, data):
    fpgaclient.write_int('mixer_cnt', lof_diff_num - 2 )  # This -2 because of the Rational a = b delay one clock for output and counter begin from 0
    time.sleep(1)
    for i in range(8):
        fpgaclient.write(subbandname+'_lo_'+str(i)+'_lo_ram', struct.pack(bramformat, *data[i::8]))
	time.sleep(1)

def read_snaps(snap_depth = 13, adc_snap_depth = 11, mixer_snap_depth = 11):
    fpga.write_int('snap1_ctrl', 0)
    fpga.write_int('adc_snap64_ctrl', 0)
    fpga.write_int('subband1_mixer_snap64_ctrl', 0)
    time.sleep(1)

    fpga.write_int('snap1_ctrl', 7)
    fpga.write_int('adc_snap64_ctrl', 7)
    fpga.write_int('subband1_mixer_snap64_ctrl', 7)
    time.sleep(1)

    pol1_re = struct.unpack('>'+str(2**snap_depth)+'l',fpga.read('snap1_bram', (2**snap_depth)*4))
    adc_data_m = struct.unpack('>'+str(2**adc_snap_depth*4)+'b',fpga.read('adc_snap64_bram_msb', (2**adc_snap_depth)*4))
    adc_data_l = struct.unpack('>'+str(2**adc_snap_depth*4)+'b',fpga.read('adc_snap64_bram_lsb', (2**adc_snap_depth)*4))
    mixer_data_m = struct.unpack('>'+str(2**mixer_snap_depth*4)+'b',fpga.read('subband1_mixer_snap64_bram_msb', (2**mixer_snap_depth)*4))
    mixer_data_l = struct.unpack('>'+str(2**mixer_snap_depth*4)+'b',fpga.read('subband1_mixer_snap64_bram_lsb', (2**mixer_snap_depth)*4))

    print size(pol1_re)
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

    pol1_re_np = array(pol1_re)
    pol1_re = pol1_re_np.astype(int8)
    #pol1 = pol1_re[0::16]
    #print pol1_re[100:200:16]
    #print pol1_re[100:150]

    return pol1_re, adc_data, mixer_data


def read_cic_snaps(fpga, dec_rate, snap_depth):
    fpga.write_int('subband1_cic1_out_ctrl', 0)
    fpga.write_int('subband1_hb_out_ctrl', 0)
    fpga.write_int('subband1_cic2_out_ctrl', 0)
    time.sleep(3)

    fpga.write_int('subband1_cic1_out_ctrl', 7)
    fpga.write_int('subband1_hb_out_ctrl', 7)
    fpga.write_int('subband1_cic2_out_ctrl', 7)
    time.sleep(3)

    fpga.write_int('subband1_cic1_out_ctrl', 0)
    fpga.write_int('subband1_hb_out_ctrl', 0)
    fpga.write_int('subband1_cic2_out_ctrl', 0)
    time.sleep(3)

    fpga.write_int('subband1_cic1_out_ctrl', 7)
    fpga.write_int('subband1_hb_out_ctrl', 7)
    fpga.write_int('subband1_cic2_out_ctrl', 7)
    time.sleep(3)

    cic1_m = struct.unpack('>'+str(2**snap_depth)+'l',fpga.read('subband1_cic1_out_bram_msb', (2**snap_depth)*4))
    cic1_l = struct.unpack('>'+str(2**snap_depth)+'L',fpga.read('subband1_cic1_out_bram_lsb', (2**snap_depth)*4))
    hb_m = struct.unpack('>'+str(2**snap_depth)+'l',fpga.read('subband1_hb_out_bram_msb', (2**snap_depth)*4))
    hb_l = struct.unpack('>'+str(2**snap_depth)+'L',fpga.read('subband1_hb_out_bram_lsb', (2**snap_depth)*4))
    cic2_m = struct.unpack('>'+str(2**snap_depth)+'l',fpga.read('subband1_cic2_out_bram_msb', (2**snap_depth)*4))
    cic2_l = struct.unpack('>'+str(2**snap_depth)+'L',fpga.read('subband1_cic2_out_bram_lsb', (2**snap_depth)*4))
    cic1_data = arange(0, size(cic1_m)/2, 1.0)
    hb_data = arange(0, size(hb_m)/2, 1.0)
    cic2_data = arange(0, size(cic2_m)/(dec_rate/8), 1.0)
    for i in range(0, size(cic1_m),2):
	cic1_data[i/2] = cic1_m[i]*(1<<10) + float(cic1_l[i])/(1<<22)
	hb_data[i/2] = hb_m[i] + float(hb_l[i])/(1<<32)
	
    for i in range(0, size(cic2_m), dec_rate/8):
	cic2_data[i/(dec_rate/8)] = cic2_m[i]*(1<<20) + float(cic2_l[i])/(1<<12)
	'''
	if cic1_m[i] <> 0:
		print 'overflow!'
		return 0, 0, 0
	if cic2_m[i] <> 0:
		print 'overflow!'
    	'''
    return cic1_data, hb_data, cic2_data	
	


def calc_lof(bandwidth,bramlength,lof_input,lof_diff_n,lof_diff_m):
    """
    Function:
        Calculate the closest frequency of lof_input, return the exact
        working frequency of the mixer, the numertor and 
        denominator of the combination.
        The available frequency combination can be: 
        1/2,1/3,1/4,1/5,2/5,1/6/,1/7,2/7,3/7,...,1/2^bramlength,2/2^bramlength,...
    Parameters:
        bandwidth: ADC working bandwidth, e.g. 600.0(MHz)
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
    #for i in range(2,2**bramlength+1):
        #for j in range(1,2**bramlength+1):
#	for j in range(1, i):
    for i in range(1, 2**bramlength):
            diff = abs(lof_input-2*bandwidth*i/(2**bramlength*1.))
            if diff < lof_diff:
                lof_diff = diff
                lof_diff_m = i
                print lof_diff,lof_diff_m,2*bandwidth*i/(2**bramlength*1.)
                if diff == 0:
                    break
    return 2*bandwidth*lof_diff_m/(2**bramlength*1.),lof_diff_m, 2**bramlength



    
