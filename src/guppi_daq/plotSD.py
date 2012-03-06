#!/opt/local/bin/python2.7
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
    pf = pyfits.open(fname)[1]
    data = pf.data.field('DATA')
    print "Found data with shape:",data.shape
    return data
    
def plotSDfile(fname,nplot=1,fadc=600.0):
    d = getSDdata(fname)
    nspec = d.shape[0]
    for pn in range(nplot):
        f,axs = plt.subplots(4,1,squeeze=True,sharex=True)
        sidx = pn*nspec/nplot + nspec/(2*nplot)
        for offs in range(4):
	    data = d[sidx,offs::4]
#	    data = data.reshape((256,4))[:,::-1].flatten() # uncomment this line for designs before r8a
#            axs[offs].plot(d[sidx,offs::4])
	    axs[offs].plot(data)
            axs[offs].text(0.1,0.99,("idx=%d:%d" % (sidx,offs)),va='top',transform=axs[offs].transAxes)
            if offs ==0:
                ax2 = plt.twiny(axs[offs])
                ax2.set_xlim(0,fadc)
                ax2.set_xlabel('MHz (assuiming %.1fMHz clk)'%fadc)
            if offs ==3:
                axs[offs].set_xlabel('bin')
            
            axs[offs].set_xlim(0,1024)
            
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
	fadc = 600.0
    plotSDfile(fn,nplot,fadc=fadc)
    plt.show()
