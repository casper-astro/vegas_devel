#!/opt/vegas/bin/python2.7

import matplotlib
#matplotlib.use('TkAgg')
from matplotlib import pyplot as plt
import pyfits
import numpy as np
import glob
import os

def getLastSD(home='.',nspectra=None):
    """ Attempt to get data from latest SD file"""
    fits = glob.glob(os.path.join(home,'vegas*.fits'))
    fits.sort()
    print "opening",fits[-1]
    data = getSDdata(fits[-1])
    if nspectra:
        data = data[-nspectra:,:]
    return data
    
def getSDdata(fname):
    pf = pyfits.open(fname)
    bt = pf[1]
    data = bt.data.field('DATA')
    cf = bt.header['OBSFREQ']
    bw = bt.header['CHAN_BW']*1024
    print "Found data with shape:",data.shape, "cf=",cf,"bw=",bw
    return data,cf,bw
    
def plotSDfile(fname,nplot=1):
    d,cf,bw = getSDdata(fname)
    nspec = d.shape[0]
    freqs = cf + bw*np.arange(1024)/1024.0  - bw/2.0
    freqs = freqs/1e6

    for pn in range(nplot):
        f,axs = plt.subplots(4,1,squeeze=True,sharex=True)
        sidx = pn*nspec/nplot + nspec/(2*nplot)
        for offs in range(4):
            data = d[sidx,offs::4]
#	    data = data.reshape((256,4))[:,::-1].flatten() # uncomment this line for designs before r8a
#            axs[offs].plot(d[sidx,offs::4])
            axs[offs].plot(freqs,data)
            axs[offs].text(0.1,0.99,("idx=%d:%d" % (sidx,offs)),va='top',transform=axs[offs].transAxes)
            if offs ==3:
                axs[offs].set_xlabel('MHz')
    f,axs = plt.subplots(4,1,squeeze=True,sharex=True)
    for offs in range(4):
        data = d[:,offs::4].mean(0)
        axs[offs].plot(freqs,data)
        axs[offs].text(0.1,0.99,("idx=%d:%d" % (sidx,offs)),va='top',transform=axs[offs].transAxes)
        if offs ==3:
            axs[offs].set_xlabel('bin')
            
if __name__ == "__main__":
    import sys
    fn = sys.argv[1]
    if len(sys.argv) >2:
        nplot = int(sys.argv[2])
    else:
        nplot = 1
    if len(sys.argv) >3:
	fadc = float(sys.argv[3])
    else:
	fadc = 1200.0
    plotSDfile(fn,nplot)
    plt.show()
