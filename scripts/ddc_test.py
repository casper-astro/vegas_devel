import scipy.io as sio

adc_data=sio.loadmat('adc_dump.mat')
adc_filter=sio.loadmat('filter_dump.mat')
adc_vals_in=sio.loadmat('adc_vals_in.mat')

#adc_vals.keys()
adc_vals=adc_data['adc_dump']
adc_filter=adc_filter['filter_dump']
adc_val=adc_values_in['adc_dump']

#vals_in = adc_vals_in[1:2048]

d = adc_vals[1:,:32768]
d2 = d.T.reshape((8*32768,))
