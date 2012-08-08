#!/usr/bin/python2.6

import numpy, corr, os, time, struct, sys, logging, pylab, matplotlib, scipy, pickle
from numpy import *
from scipy import *

execfile('mixer_funcs.py')

katcp_port = 7147
roach_name = '192.168.40.67'

##################################################################
# Parameter Setting
##################################################################

# ADC working bandwidth (MHz)
bandwidth = 200.0

# Decimation Rate
dec_rate = 64

# Mixer BRAM length (2^?)
bramlength = 10

# Mixer writer format string
bramformat = '>'+str(2**bramlength)+'I'

# Input LO frequency (MHz)
#lof_input = float((512+16)/1024.0)*1600.0
lof_input = 101

# Input signal frequency (MHz)
signal_input = 400

# Available LO frequency list
lof_avail = []
for i in range(2,2**bramlength):
    lof_avail.append(2*bandwidth/(i*1.))

# The closest frequency of the lof_input
lof_output = 0

lof_diff_n = 0
lof_diff_m = 0
lof_diff_num = 0

print lof_input
lof_output,lof_diff_n,lof_diff_m = calc_lof(bandwidth,bramlength,lof_input, lof_diff_n,lof_diff_m)

lof_diff_num = lof_diff_m

#lof_diff_num = 2048 + 2

print 'lof_output: %f'%lof_output
print 'lof_diff_num: %d'%lof_diff_num

# Snap depth
snap_depth = 13
adc_snap_depth = 13
mixer_snap_depth = 11

# LO frequency for computing
lof = lof_output




if __name__ == '__main__':
#roach03 is recorded in /etc/hosts file, or use IP address
    #roach = 'roach03'
    loggers = []
    lh = corr.log_handlers.DebugLogHandler()
    logger = logging.getLogger(roach_name)
    logger.addHandler(lh)
    logger.setLevel(10)

#connect roach
    fpga = corr.katcp_wrapper.FpgaClient(roach_name, katcp_port, timeout=10, logger=logger)
    time.sleep(1)

    if fpga.is_connected():
        print 'Connected\n'
    else:
        print 'ERROR connecting to roach'
        exit_fail()

    lo_input = wave_gen(lof, bandwidth*2, 2**bramlength*8)
    '''
    lo_input = constant_wave_gen(2**bramlength*8)
    print 'Generating constant wave (DC) in the BRAM...'
    lof = 0
    '''

    fill_mixer_bram(fpga, 'subband1', bramformat, lo_input)
    time.sleep(3)
    set_gain_sg(fpga, '%c%c%c%c'%(0x00, 0x01, 0x00, 0x00))
    print 'Setting gain to: 0x00, 0x01, 0x00, 0x00'
    time.sleep(1)

    pol1_re, adc_data, mixer_data = read_snaps(snap_depth)
    print size(pol1_re)
    print pol1_re[100:200:(dec_rate/8)]
    print pol1_re[100:120]
    pol1_re = pol1_re[0::(dec_rate/8)]

#plot it

    x = []
    a = -bandwidth/dec_rate
    for i in range(size(pol1_re)):
        x.append(a)
        a = a + 2*(bandwidth/dec_rate)/size(pol1_re)

    pylab.ion()
    pylab.figure()

   # while(1):
    if 1 <> 2:
	print 'lo frequency (MHz):' + str(lof_output)
	#os.system('./sg_ctrl freq')

        pylab.subplot(221)
        pylab.title('One sin cycle in BRAM')
        pylab.xlabel('N')
        #pylab.plot(lo_input[0:lof_diff_num+1],'-o')
	pylab.plot(adc_data[100:500], '-o')
        pylab.hold(False)

        pylab.subplot(222)
        #pylab.title('Signal FFT')
        #pylab.xlabel('Frequency: MHz')
       # pylab.plot(x,abs(pylab.fftshift(fft(adc_data))))
        pylab.plot(mixer_data[100:200], '-o')
	#pylab.semilogy(abs(fft(mixer_data)))
	pylab.hold(False)

        pylab.subplot(223)
        pylab.title('Available LO frequencies')
        pylab.ylabel('MHz')
        #pylab.plot(lof_avail,'-o')
	pylab.plot(pol1_re[100:200], '-o')
        pylab.hold(False)

        pol1_re_fft = fft(pol1_re)
        pylab.subplot(224)
        pylab.title('Mixed FFT,  Signal: '+str(signal_input)+'MHz,  LO: '+str(lof_output)+'MHz')
        pylab.xlabel('Frequency: MHz')
        pylab.semilogy(x,abs(pylab.fftshift((pol1_re_fft))))

        pylab.hold(False)
        pylab.draw()

        pol1_re = read_snaps(snap_depth)
	pol1_re = pol1_re[0::(dec_rate/8)]

        time.sleep(2)

    #cic1_data, hb_data, cic2_data = read_cic_snaps(fpga, dec_rate, snap_depth)
    #pylab.figure()
    #pylab.subplot(221)
    #pylab.semilogy(abs(pylab.fftshift(fft(cic1_data[0:1024]))))
    #pylab.title('FFT of the output of CIC_1')
    #pylab.subplot(222)
    #pylab.semilogy(abs(pylab.fftshift(fft(hb_data[0:1024]))))
    #pylab.title('FFT of the output of Halfband filter')
    #pylab.subplot(223)
    #pylab.semilogy(abs(pylab.fftshift(fft(cic2_data))))
    #pylab.title('FFT of the output of the second CIC')
    #pylab.subplot(224)
    #pylab.semilogy(abs(pylab.fftshift(fft(adc_data[0:1024]))))
    #pylab.title('FFT of the output of ADC')
    #pylab.figure()
    #pylab.subplot(221)
    #pylab.plot(cic1_data[100:200],'-o')
    #pylab.title('Output of the first CIC')
    #pylab.subplot(222)
    #pylab.plot(hb_data[100:200], '-o')
    #pylab.title('Output of the Halfband filter')
    #pylab.subplot(223)
    #pylab.plot(cic2_data[100:150],'-o')
    #pylab.title('Output of the second CIC')
    #pylab.subplot(224)
    pylab.plot(adc_data[100:150], '-o')
    pylab.title('Output of ADC')
    pylab.figure()
    #pylab.semilogy(abs(pylab.fftshift(fft(cic1_data))))

