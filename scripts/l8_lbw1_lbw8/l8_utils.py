import numpy as np

def lo_setup(fpga,mixer,lo_freq,fs,bram_len=10,n_inputs=16):
    wave_len = n_inputs * 2**bram_len  # This is the number of complex samples in the waveform
    frac_lo = int(np.round((lo_freq/fs)*wave_len))
    actual_lo = fs*frac_lo/float(wave_len)
    wave_data = np.zeros((wave_len*2,),dtype='>i2')  #len * 2 for real and imag
    wave_dataf = np.zeros((wave_len*2,),dtype='float')  #len * 2 for real and imag for debugging only
    if lo_freq == 0:
        wave_data[:] = 2**15 - 1 # set values to maximum constant value to pass wave directly through mixer
    else:
        wave_dataf[::2] = np.sin(2*np.pi*frac_lo*np.arange(wave_len)/float(wave_len))*(2**15-1)  #-1 to ensure we never hit 2**15 which would translate to -2**15 and cause a glitch
        wave_dataf[1::2] = np.cos(2*np.pi*frac_lo*np.arange(wave_len)/float(wave_len))*(2**15-1)
        wave_data[:] = np.round(wave_dataf).astype('>i2')
    
    fpga.write_int(('%s_mixer_cnt' % mixer),2**bram_len -2)
    for k in range(n_inputs):
        bram_name = '%s_lo_%d_lo_ram' % (mixer,k)
        fpga.write(bram_name,wave_data[k::n_inputs].tostring())
    return actual_lo,wave_data
