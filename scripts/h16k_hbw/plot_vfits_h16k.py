#!/opt/vegas/bin/python

import matplotlib
from matplotlib import pyplot as plt
import pyfits
import numpy as np
import glob
import os
import sys

fn = sys.argv[1]

hdu = pyfits.open(fn)[1]
nchan = hdu.header['NCHAN']
nsb = hdu.header['NSUBBAND']
nstokes = 4
specnum = 0
#print nchan, nsb, nstokes
d = hdu.data[specnum]['DATA']
ns = d.shape[0]
print ns

d = d.reshape((1, nchan, nsb, nstokes))

#d[:,0,:,:] = d[:,1,:,:]
#d[:,4095,:,:] = d[:,4094,:,:]
#d = np.fft.fftshift(d, axes=1)
#print d.shape
clk = 1500.0
f_lo = 750.0
bw=2*f_lo
f = np.linspace(f_lo - bw/2., f_lo + bw/2., nchan)

nsb = 1;
for i in range(nsb):
    for j in range(nstokes):
        plt.subplot(nstokes, nsb, (j*nsb)+(i+1))
        plt.plot(f, 10 * np.log10(d[0,:,i,j]))
        #plt.plot(f, d[0,:,i,j])
        plt.xlabel('f (MHz)')


#plt.plot(f, 10 * np.log10(d[0,:,1,0]))
#plt.plot(f, d[0,:,0,1])
plt.show()

