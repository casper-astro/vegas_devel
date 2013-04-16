import os,sys
from katcp_wrapper import FpgaClient
from matplotlib import pyplot as plt
import time
import Pyro4
import guppi_utils
from parfile import psr_par

boffile = 'osf4rddc8t9fx8ka101_2013_Feb_21_1416.bof'
ips = range(101,110)  #gpu1-gpu9 have final octet 101,102,etc
ipbase = (10<<24)+(17<<16)

segdict = {327 : [0,1,2,3,6,7,4,5],
           430 : [0,1,2,3,6,7,4,5],
           1400 : [4,5,6,7,0,1,2,3]
           }
nyqdict = {327 : 2,
           430 : 2,
           1400 : 3,}
feparams = {327 : dict(FRONTEND='327', FD_POLN='LIN'),
            430 : dict(FRONTEND='430', FD_POLN='CIRC'),
            1400 : dict(FRONTEND='L-wide',FD_POLN='LIN')}
freqoffs = {327 : -427,
            430 : -327,
            1400 : 180}
portreg = {0:'dist_port01',
           1:'dist_port01',
           2:'dist_port23',
           3:'dist_port23',
           4:'dist_port45',
           5:'dist_port45',
           6:'dist_port67',
           7:'dist_port67',
           }


if not globals().has_key('r1'):
    r1 = FpgaClient('roach')
    
gpud = {}
def getServers():
    global gpud
    for k in [1,2,3,5,6,7,8,9]:
        if not globals().has_key('g%d' % k):
            globals()['g%d' % k] = Pyro4.Proxy("PYRONAME:guppiServer%d" % k)
        gpud[k] = globals()['g%d' % k]
        gpud[k]._pyroReconnect()
getServers()
gpulist = gpud.values()

gpus = [9]

#gs = guppi_utils.guppi_status()

def setStartTime(t,gpus=gpus):
    import astro_utils 
    t = int(t)
    mjd = astro_utils.current_MJD(t) # we have modified this to take an argument of unix epoch optionally
    mjdi = int(np.floor(mjd))
    mjdf = int(np.fmod(mjd,1)*86400)
    for gpu in gpus:
        g = gpud[gpu]
        g.setParams(STT_IMJD=mjdi, STT_SMJD=mjdf)
        
def updateScanNum(gpus=gpus):        
    try:
        g = gpud[gpus[0]]
        sn = g.getParam('SCANNUM')['SCANNUM']
    except:
        print "WARNING: could not find previous scan number, resetting to 1!"
        sn = 0
    sn += 1
    setParams(gpus,SCANNUM=sn)
    return sn
    
def startRawCal(gpus=gpus,scanlen=None):
    startRaw(gpus=gpus,cal="ON",scanlen=scanlen)
    
def startRaw(gpus=gpus,cal = "OFF",scanlen=None):
    sn = updateScanNum(gpus)
    t = int(time.time())
    startat = t + 5 # startat is even pps tick
    armat = startat-0.6 # armat is when we arm the roach
    print "initializing scan %d planning to start at %d" % (sn,startat)
    setStartTime(startat, gpus) 
    for gpu in gpus:
        g = gpud[gpu]
        g.csCommand('STOP')
        g.setParams(OBS_MODE="RAW",
                    CAL_MODE=cal,
                    SCANLEN=90.0,
                    )
        g.csCommand('START')
    time.sleep(2)
    for gpu in gpus:
        g = gpud[gpu]
        stat = g.getParam("NETSTAT")['NETSTAT']
        print gpu,":",stat
    print "syncing... have %.2f seconds to arm"  % (armat-time.time())
    sync(at = armat)
    time.sleep(2)
    for gpu in gpus:
        g = gpud[gpu]
        stt = g.getParam('STTVALID')['STTVALID']
        stat = g.getParam("NETSTAT")['NETSTAT']
        if stt == 0:
            print "ERROR: STTVALID did not go true for gpu ",gpu,"netstat is:",stat
    if scanlen:
        print "Waiting for scan to finish... press ctrl-c if you want to stop scan yourself later"
        endat = startat + scanlen
        print endat
        while time.time() < endat:
            print ("\r%7.1f seconds remaining" % (endat - time.time())),
            sys.stdout.flush()
            time.sleep(0.1)
        print "stopping"
        killAll(gpus)

def autoAtten(goal=20.0):
    v = r1.read_int('adc_ctrl')
    if (v & 0x8080) != 0x8080:
        print "Found adc inputs off, initializing attenuators to 0 dB"
        r1.write_int('adc_ctrl',0x8080)
        xatt = 0
        yatt = 0
    else:
        xatt = v & 0x3F
        yatt = (v >> 8) & 0x3F
    tries = 0
    while tries < 5:
        x,y = getAdc()
        x = x.std()
        y = y.std()
        xerr = 20*np.log10(x/goal)
        yerr = 20*np.log10(y/goal)
        print "X @ %.1f dB: %.1f  %.1f dB  || Y @ %.1f dB: %.1f %.1f dB" % (xatt/2.0,x,xerr,yatt/2.0,y,yerr)
        if np.abs(xerr) < 2.0 and np.abs(yerr) < 2.0:
            print "finished"
            break
        xatt = int(np.round(xatt + xerr*2.0))
        if xatt > 63:
            xatt = 63
        if xatt < 0:
            xatt = 0
        yatt = int(np.round(yatt + yerr*2.0))
        if yatt > 63:
            yatt = 63
        if yatt < 0:
            yatt = 0
        r1.write_int('adc_ctrl', 0x8080 + (yatt<<8) + xatt)
        tries +=1
    else:
        print "Failed to converge!"
        return
    print "Success!"

def setAdcClock(frq,extref=True,chan_spacing=2.0):
    import valon_eth_lib
    vs = valon_eth_lib.Synthesizer('vclk',6001,2.0)
    reffreq = vs.get_reference()
    if reffreq != 10*1000*1000:
        print "Error! frequency reference is not 10 MHz!!"
        vs.conn.close()
        return
    ref = vs.get_ref_select()
    if extref and (ref==0):
        print "switching from internal to external reference"
        vs.set_ref_select(1)
    elif (not extref) and (ref==1):
        print "switching from external to internal reference"
        vs.set_ref_select(0)
    print "Current frequencies: A:",vs.get_frequency(0),"B:", vs.get_frequency(8)
    print "Power levels A:",vs.get_power(0),"B:",vs.get_power(8)
    tries = 0
    while tries < 5:
        vs.set_frequency(0,frq,chan_spacing=chan_spacing)
        vs.set_frequency(8,frq,chan_spacing=chan_spacing)
        time.sleep(0.5)
        lockA = vs.get_phase_lock(0)
        lockB = vs.get_phase_lock(8)
        if lockA and lockB:
            time.sleep(1)
            lockA = vs.get_phase_lock(0)
            lockB = vs.get_phase_lock(8)
            if lockA and lockB:
                print "Synthesizers locked"
                print "Current frequencies: A:",vs.get_frequency(0),"B:", vs.get_frequency(8)
                try:
                    print "FPGA clock:",r1.est_brd_clk()
                except:
                    print "Couldn't read FPGA clock"
                break
        print "synths didn't lock, trying again. A:",lockA,"B:",lockB
        vs.set_frequency(0,990.0,chan_spacing=chan_spacing)
        vs.set_frequency(8,990.0,chan_spacing=chan_spacing)
    else:
        print "failed after 5 tries"
        vs.set_frequency(0,frq,chan_spacing=chan_spacing)
        vs.set_frequency(8,frq,chan_spacing=chan_spacing)
    vs.conn.close()


def setParams(nodes=gpus,**kwargs):
    for node in nodes:
        gpud[node].setParams(**kwargs)
    
    
def plotGpu(gpun,Fs=1024.,rf=True,ndat=2**23, x16=False):
    gpu = gpud[gpun]
    s = gpu.getParam('SEGMENT')['SEGMENT']
    
    fe = gpu.getParam('FRONTEND')['FRONTEND']
    for k,v in feparams.items():
        if v['FRONTEND'] == fe:
            band = k
    nyq = nyqdict[band]
    if rf:
        offset = freqoffs[band]
    else:
        offset = 0
#    d = rawData(gpu.getData(0,ndat))
    chwid = 2*Fs/8.0/512.0
    fc,chans = segFreq(s,Fs=Fs,nyq=nyq,x16=x16)
    fc = fc + offset    
    if nyq == 2:
        sign = -1
    else:
        sign = 1
    nch = fc.shape[0]
    d = getGpuData(gpun, ndat,nchan=nch)
    f = plt.figure()
    ax1 = f.add_subplot(211)
    ax2 = f.add_subplot(212)
    ax1.plot(d[:,:,0].real.max(0),'--b')
    ax1.plot(d[:,:,1].real.max(0),'--r')
    ax1.plot(d[:,:,0].real.min(0),'--b')
    ax1.plot(d[:,:,1].real.min(0),'--r')
    ax1.plot(d[:,:,0].real.std(0),'b')
    ax1.plot(d[:,:,1].real.std(0),'r')
    f.suptitle('gpu %d segment %d' % (gpun,s))
    

    for ch in range(d.shape[1]):
        Pxx,freqs = mlab.psd(d[:,ch,0],NFFT=128,Fs=chwid)
        Pyy,freqs = mlab.psd(d[:,ch,1],NFFT=128,Fs=chwid)
        ax2.plot(sign*freqs + fc[ch], 10*np.log10(Pxx),'b')
        ax2.plot(sign*freqs + fc[ch], 10*np.log10(Pyy),'r')
    
    
def plotAdc(Fs=1024.0, lo=-2):
    x,y = getAdc()
    f = plt.figure()
    ax1 = f.add_subplot(311)
    ax2 = f.add_subplot(312)
    ax3 = f.add_subplot(313)
    hx,b = np.histogram(x,bins=np.arange(-128,128))
    hy,b = np.histogram(y,bins=np.arange(-128,128))
    label = 'ADC0: ptp=%.0f, rms=%.1f, %.1f dB' % (x.ptp(),x.std(),20*np.log10(x.std()/20.0))
    ax1.semilogy(np.arange(-127,128),hx,lw=2,label=label,color='b')
    label = 'ADC1: ptp=%.0f, rms=%.1f, %.1f dB' % (y.ptp(),y.std(),20*np.log10(y.std()/20.0))
    ax1.semilogy(np.arange(-127,128),hy,lw=2,label=label,color='r')
    ax1.set_xlim(-128,128)
    ax1.set_ylim(1e-3,1e3)
    ax1.legend(loc='upper right',prop=dict(size='x-small'))
    ax2.psd(x,NFFT=256,Fs=Fs,lw=1.5,label='ADC0',color='b')
    ax2.psd(y,NFFT=256,Fs=Fs,lw=1.5,label='ADC1',color='r')
    ax3.plot(x[:300],'b',lw=1.5)
    ax3.plot(y[:300],'r',lw=1.5)
    ax3.set_ylim(-128,128)
    
    dx,dy = getDdc()
    Pxx,freqs = plt.mlab.psd(dx,NFFT=512,Fs=Fs/8.0)
    Pyy,freqs = plt.mlab.psd(dy,NFFT=512,Fs=Fs/8.0)
    offd = {-1:.25, -2:.5}
    freqs = freqs + offd[lo] * Fs/2.0
    f = plt.figure()
    ax1 = f.add_subplot(411)
    ax2 = f.add_subplot(412)
    ax2b = f.add_subplot(413,sharex=ax2)
    ax3 = f.add_subplot(414)
    hx,b = np.histogram(dx.real/2.0**15,bins=200)
    hy,b = np.histogram(dy.real/2.0**15,bins=200)
    ax1.semilogy(b[:-1],hx,lw=2,color='b')
    ax1.semilogy(b[:-1],hy,lw=2,color='r')
    ax2.plot(freqs,10*np.log10(Pxx),lw=1.5,color='b')
    ax2.plot(freqs,10*np.log10(Pyy),lw=1.5,color='r')
    ax2.grid(True)
    ax3.plot(dx.real[:300],'b',lw=1.5)
    ax3.plot(dy.real[:300],'r',lw=1.5)
    
    fx,fy = getFft()
    ax2.plot(freqs,20*np.log10(np.abs(np.fft.fftshift(fx[::2]))),color='k')
    d = getDout()
    trans = plt.matplotlib.transforms.blended_transform_factory(ax2.transData, ax2.transAxes)
    transb = plt.matplotlib.transforms.blended_transform_factory(ax2b.transData, ax2b.transAxes)
    for r in range(4):
        gpua = (r1.read_int('dist_ip%d' % (2*r)) & 0xFF) - 100
        gpub = (r1.read_int('dist_ip%d' % (2*r+1)) & 0xFF) - 100
        if gpua < 0:
            gpua = 999
        if gpub < 0:
            gpub = 999
        fsh = np.fft.fftshift(freqs)
        nf = len(freqs)
        fr = fsh[r*nf/4:(r+1)*nf/4]
        if r % 2:
            color = 'g'
        else:
            color = 'm'
        rect = plt.Rectangle((fr[0],0),(fr[-1]-fr[0]),1, transform=trans, alpha=0.2, color=color)
        ax2.add_patch(rect)
        print (fr[-1]-fr[0])/2.0
        print ("%d : E %d  O %d" % (r,gpua,gpub))
        ax2.text((fr[-1]-fr[0])/2.0 + fr[0],0.87,("%d : E %d  O %d" % (r,gpua,gpub)),transform=trans,fontdict=dict(size='small'),ha='center',va='bottom')
        ax2b.plot(fr[::2],20*np.log10(np.abs(d[:,r*2,0,0])),'r')
        ax2b.plot(fr[1::2],20*np.log10(np.abs(d[:,r*2+1,0,0])),'b')
        rect = plt.Rectangle((fr[0],0),(fr[-1]-fr[0]),1, transform=transb, alpha=0.2, color=color)
        ax2b.add_patch(rect)
#    ax2.plot(freqs,20*np.log10(np.abs(np.fft.fftshift(fy[::2]))),color='r')

def getParam(name):
    for n,g in gpud.items():
        try:
            print n,":",g.getParam(name)
        except:
            print n,": failed"
            
def monitor():
    for g in gpud.values():
        g.csCommand('MONITOR')
                
def restartDaq():
    for n,g in gpud.items():
        g.restartDaq()
def getStatus():
    s = r1.read_int('dist_tgestatus')
    print '%08x' % s
    print '%08x' % (s & 0x1F)
    print '%08x' % ((s>>5) & 0x1F)
    b = 0
    for k in range(1,-1,-1):
        print "tge%d" % k
        print "----"
        if s & (1<<b):
            print "**** TX OVERFLOW ****"
        b += 1
        if s & (1<<b):
            print "**** TX AFULL ****"
        b += 1
        if s & (1<<b):
            print "TX ",
        else:
            print "   ",
        b += 1
        if s & (1<<b):
            print "RX ",
        else:
            print "   ",
        b += 1
        if s & (1<<b):
            print "UP"
        else:
            print "DOWN"
        b += 1
        print "============"
    try:
        getParam("NETSTAT")
    except:
        pass
        
        
def startDspsr(cmdline,gpus=gpus,dosync=True):
    print "using gpus:",gpus
    print "stopping daqs"
    sys.stdout.flush()
    for gpu in gpus:
        g = gpud[gpu]
        g.endProc()
        g.csCommand('STOP')
    sync()
    time.sleep(1)
    print "running", cmdline
    sys.stdout.flush()
    for gpu in gpus:
        g = gpud[gpu]
        g.startProc(cmdline)
        g.csCommand('START')
    if dosync:
        time.sleep(2)
        print "syncing.."
        sys.stdout.flush()
        sync()
        
def killAll(gpus=gpus):
    print "stopping daqs"
    sys.stdout.flush()
    for gpu in gpus:
        g = gpud[gpu]
        g.endProc()
        g.csCommand('STOP')
        
def setupObs(parfile, band, gpus = gpus, Fs=1024.0, nchan=64):
    if nchan == 32:
        x16 = True
    else:
        x16 = False
    generalParams = getObsParams(parfile)
    segs = segdict[band][:len(gpus)] # get the most important segments given the number of gpus available
    for gpu in gpus:
        if not gpud.has_key(gpu):
            gpud[gpu] = Pyro4.Proxy("PYRONAME:guppiServer%d" % gpu)
    import astro_utils
    mjd = astro_utils.current_MJD()
    mjdi = int(np.floor(mjd))
    mjds = int(np.round(np.fmod(mjd,1)*86400))
    
    for k in range(8):
        r1.write_int(('dist_ip%d' % k), ipbase + ips[3])
            
    nyq = nyqdict[band]
    for gpu,seg in zip(gpus,segs):
        print "assigning gpu",gpu,"to segment",seg
        g = gpud[gpu]
        frqs,chans = segFreq(seg, Fs=Fs, nyq=nyq,x16=x16)
        frqs = frqs + freqoffs[band]
        fc = frqs[nchan/2]
        chanbw = frqs[1]-frqs[0]  # check if this should be signed or not
        bw = chanbw * nchan
        tbin = np.abs(1e-6/chanbw)
        
        if seg % 2:
            roachip = '10.17.0.221'
        else:
            roachip = '10.17.0.220'
        port = 61000 + seg
        
        datadir = '/data/gpu/partial/gpu0%d/cspuppi' % gpu
        
        g.setParams(**generalParams)
        g.setParams(**(feparams[band]))
        g.setParams(TBIN=tbin,
                    CHAN_BW=chanbw,
                    OBSFREQ = fc,
                    OBSBW = bw,
                    OBSNCHAN=nchan,
                    SEGMENT=seg,
                    DATAHOST=roachip,
                    DATAPORT=port,
                    DATADIR=datadir,
                    STT_IMJD=mjdi,
                    STT_SMJD=mjds,
                    STTVALID=0,
                    NETSTAT='exiting')
        r1.write_int(('dist_ip%d' % seg), ipbase + ips[gpu-1])
        pp = r1.read_uint(portreg[seg])
        if (seg % 2) == 0:
            pp = (pp & 0xFFFF) | (port<<16)
        else:
            pp = (pp & 0xFFFF0000) | port
        r1.write_int(portreg[seg],pp)

def autoLevel(gpus=gpus, goal=20.0,nchan=512,init_gain=128,x16=False):
    stopData()
    time.sleep(1)
    sync() # ensure data is flowing
    getStatus()
    print "initializing gains"
    r1.write_int('pol0_gain',init_gain)
    r1.write_int('pol1_gain',init_gain)
    for ch in range(nchan):
        r1.write_int('pol0_addr',ch)
        r1.write_int('pol1_addr',ch)

    for gpu in gpus:  
        print "starting gpu", gpu      
        g = gpud[gpu]
        g.csCommand('STOP')
        g.csCommand('MONITOR')

    print "waiting for new data"
    sys.stdout.flush()
    time.sleep(4)


    print "pass 1"
    gains = {}
    settings = []
    for gpu in gpus:
        g = gpud[gpu]
        print "equalizing gpu", gpu
        s = g.getParam('SEGMENT')['SEGMENT']
        frqs,chans = segFreq(s, x16=x16)
#        chunk = s/2
#        half = 1-(s % 2)
#        chans = range(nchan)[chunk*nchan/4:(chunk+1)*nchan/4][half::2]
        
        d = getGpuData(gpu, 2**20, nchan=chans.shape[0]) #rawData(g.getData(0,2**23))
        rms = d.std(0)
#        print zip(chans,rms[:,0])
        pol0 = np.round(init_gain*(goal/rms[:,0])).clip(0,2**18-1).astype('int')
        pol1 = np.round(init_gain*(goal/rms[:,1])).clip(0,2**18-1).astype('int')
#        print pol0
        for n,ch in enumerate(chans):
            settings.append((ch,pol0[n],pol1[n]))
#            r1.write_int('pol0_gain',pol0[n])
#            r1.write_int('pol0_addr',ch)
#            print pol0[n],ch
#            r1.write_int('pol1_gain',pol1[n])
#            r1.write_int('pol1_addr',ch)
        gains[gpu] = (pol0,pol1)
        
    r1.write_int('pol0_addr',0)
    r1.write_int('pol1_addr',0)
    for ch,p0,p1 in settings:
        r1.write_int('pol0_addr',ch)
        r1.write_int('pol1_addr',ch)
        r1.write_int('pol0_gain',p0)
        r1.write_int('pol1_gain',p1)
#    print settings
    time.sleep(2)
    print "pass 2"
    settings = []
    for gpu in gpus:
        g = gpud[gpu]
        print "equalizing gpu", gpu
        s = g.getParam('SEGMENT')['SEGMENT']
        frqs,chans = segFreq(s, x16=x16)
#        chunk = s/2
#        half = 1-(s % 2)
#        chans = range(nchan)[chunk*nchan/4:(chunk+1)*nchan/4][half::2]
        
        d = getGpuData(gpu, 2**20, nchan=chans.shape[0]) #rawData(g.getData(0,2**23))
        rms = d.std(0)
#        print rms[:,0]
#        print chans
        pol0,pol1 = gains[gpu]
        pol0 = np.round(pol0*(goal/rms[:,0])).clip(0,2**18-1).astype('int')
        pol1 = np.round(pol1*(goal/rms[:,1])).clip(0,2**18-1).astype('int')
        #print pol0
        for n,ch in enumerate(chans):
            settings.append((ch,pol0[n],pol1[n]))
#            r1.write_int('pol0_gain',pol0[n])
#            r1.write_int('pol0_addr',ch)
#            print pol0[n],ch
#            r1.write_int('pol1_gain',pol1[n])
#            r1.write_int('pol1_addr',ch)
        gains[gpu] = (pol0,pol1)
    r1.write_int('pol0_addr',0)
    r1.write_int('pol1_addr',0)
    for ch,p0,p1 in settings:
        r1.write_int('pol0_addr',ch)
        r1.write_int('pol1_addr',ch)
        r1.write_int('pol0_gain',p0)
        r1.write_int('pol1_gain',p1)
#    print settings
#    return gains

def resetGains(init_gain=128,nchan=512):
    r1.write_int('pol0_gain',init_gain)
    r1.write_int('pol1_gain',init_gain)
    for ch in range(nchan):
        r1.write_int('pol0_addr',ch)
        r1.write_int('pol1_addr',ch)


def init(reprog=True,starttap=True,boffile=boffile):
    if reprog:
        print "reprogramming the roach with %s ..." % boffile
        r1.progdev('')
        print r1.progdev(boffile)
    print "checking roach clock..."
    print r1.est_brd_clk()
    r1.write_int('fftshift',0x55555)
    if starttap:
        print "starting tap0"
        r1.tap_start('tap0','dist_tge0',0x020200001122,(10<<24)+(17<<16)+220,63000)
        print "starting tap1"
        r1.tap_start('tap1','dist_tge1',0x020200001123,(10<<24)+(17<<16)+221,63000)
        
def killServers(gpus=gpus):
    for gpu in gpus:
        os.system('ssh gpu0%d "pkill -f osfServer"' % gpu)       
        
def startServers(gpus=gpus):
    for gpu in gpus:
        os.system('ssh -nx gpu0%d "source /home/gpu/gjones/puppi.sh; nohup python /home/gpu/gjones/osfServer.py </dev/null &> /home/gpu/gjones/logs/server%d.log &"' % (gpu,gpu,gpu))

def getGpuData(gpu,ndat=2**23, nchan=64):    
    g = gpud[gpu]
    blk = g.getParam('CURBLOCK')['CURBLOCK']
    blk = blk - 1
    if blk < 0:
        blk = 7
    return rawData(g.getData(blk,ndat), nchan=nchan)

def getAdc():
    raw = r1.snapshot_get('adcsnap',man_trig=True,man_valid=True)['data']
    if len(raw) == 1024*8:
        d = np.fromstring(raw,dtype='>i4')
    else:
        d = np.fromstring(raw,dtype='>i8')
    x = d[::2].copy().view('int8')
    y = d[1::2].copy().view('int8')
    return x,y
    
def getDdc():
    raw = r1.snapshot_get('pol0_ddcout',man_trig=True,man_valid=True)['data']
    dx = np.fromstring(raw,dtype='>i2').astype('float').view('complex')
    raw = r1.snapshot_get('pol1_ddcout',man_trig=True,man_valid=True)['data']
    dy = np.fromstring(raw,dtype='>i2').astype('float').view('complex')
    return dx,dy

def getFft():
    dx = np.fromstring(r1.read('pol0_fftout_bram',4096),dtype='>i2').astype('float').view('complex')
    dy = np.fromstring(r1.read('pol1_fftout_bram',4096),dtype='>i2').astype('float').view('complex')
    return dx,dy

def getDout():
    d = np.fromstring(r1.read('dout_bram',4096),dtype='int8').astype('float').view('complex').reshape((64,8,2,2))
    return d
    

        
    
def tvgData(din):
    din = din.view('uint8')
    d16 = din[::2].astype('int') * 256 + din[1::2].astype('int')
    return d16

def rawData(din, nchan=64,flip=0):
    din = din.astype('float32').view('complex64')
#    ntime = din.shape[0]/(nchan*2*2)
#    din = din.reshape((ntime,nchan,2,2))
#    dout = np.empty((ntime*2,nchan,2),dtype='complex64')
#    dout[::2,:,:] = din[:,:,:,flip]
#    dout[1::2,:,:] = din[:,:,:,1-flip]
#    return dout
    ntime = din.shape[0]/(nchan*2)
    return din.reshape((ntime,nchan,2))
    
def segFreq(s,Fs=1024.0,nyq=2,x16 = False):
    chwid = 2*Fs/8.0/512.0
    nch = 512
    if nyq == 2:
        fcs = Fs - (Fs/4.0 + np.fft.fftshift((chwid/2.0)*(np.arange(nch)-nch/2)))
    elif nyq == 3:
        fcs = Fs + (Fs/4.0 + np.fft.fftshift((chwid/2.0)*(np.arange(nch)-nch/2)))
    if not x16:
        #original x8 distributor case
        chunk = s/2
        half = s%2
        chans = range(chunk*nch/4, (chunk+1)*nch/4)[half::2]
        return fcs[chans], np.array(chans,dtype='int') #[chunk*nch/4:(chunk+1)*nch/4][half::2]
    else:
        if s <  4:
            #first 4 segments are not remapped
            pass
        elif s >=8:
            raise Exception("Non existant segments are not handled")
        else:
            # last 4 segments are remmaped to end
            s = s + 8   
        chunk = s/2
        half = s%2
        chans = range(chunk*nch/8, (chunk+1)*nch/8)[half::2]
        return fcs[chans], np.array(chans,dtype='int') #[chunk*nch/8:(chunk+1)*nch/8][half::2]

    
def setupNet():
    """
    This isn't much used anymore
    """
    port = 61000
    r1.write_int('dist_port01',((port+1)<<16) + port+1)
    r1.write_int('dist_port23',((port+0)<<16) + port+3)
    r1.write_int('dist_port45',((port+4)<<16) + port+5)
    r1.write_int('dist_port67',((port+6)<<16) + port+7)
    r1.write_int('dist_ip0', ipbase + ips[7])
    r1.write_int('dist_ip1', ipbase + ips[7])
    r1.write_int('dist_ip2', ipbase + ips[8])
    r1.write_int('dist_ip3', ipbase + ips[7])
    r1.write_int('dist_ip4', ipbase + ips[7])
    r1.write_int('dist_ip5', ipbase + ips[7])
    r1.write_int('dist_ip6', ipbase + ips[7])
    r1.write_int('dist_ip7', ipbase + ips[7])
    
def stopData():
    r1.write_int('ctrl',0)
    r1.write_int('ctrl',2)
    r1.write_int('ctrl',6)
    r1.write_int('ctrl',2)

        
        


def getObsParams(parfile):
    if not os.path.exists(parfile):
        print "No such parfile:", parfile
        parfile = '/home/gpu/tzpar/' + parfile + '.par'
        if not os.path.exists(parfile):
            print "Could not find parfile:", parfile
            return
    print "using: ", parfile
    par = psr_par(parfile)
    # Get source name
    try:
        psr = par.PSRJ
    except:
        try:
            psr = par.PSR
        except:
            print "Could not get source from parfile"
            return
    
    # Get position
    try:
        ra = par.RAJ
        dec = par.DECJ
    except:
        try:
            ra = par.RA
            dec = par.DEC
        except:
            print "Error reading RA/DEC from parfile."
            return
        
#    frontend = "L-wide"
#    poln = "LIN"
#    calmode = "OFF"
#    scanlen=3600.0
#    scannum = 992
#    bw = 200.0
#    chbw = bw/512
#    bwpernode = bw/8
#    tbin = 1/chbw
    common_params = {
                     "SRC_NAME": psr,
                     "OBSERVER": "gjones",
                     "RA_STR": ra,
                     "DEC_STR": dec,
                     "TELESCOP": "Arecibo",
#                     "FRONTEND": frontend,
                     "PROJID" : "P2721",
#                     "FD_POLN": poln,
                     "TRK_MODE": "TRACK",
                     "CAL_MODE": "OFF",
#                     "SCANLEN" : scanlen,
                     "BACKEND" : "CSPUPPI",
                     "PKTFMT" : "SIMPLE",
                     "POL_TYPE" : "IQUV",
                     "CAL_FREQ" : 25.0,
                     "CAL_DCYC" : 0.5,
                     "CAL_PHS" : 0.0,
#                     "OBS_MODE" : "RAW",
                     "NPOL" : 4,
                     "NBITS" : 8,
                     "PFB_OVER" : 4,
                     "NBITSADC" : 8,
                     "ACC_LEN" : 1,
                     "NRCVR" : 2,
                     "ONLY_I" : 0,
                     "DS_TIME" : 1,
                     "DS_FREQ" : 1,
                     "TFOLD" : 10.0,
                     "NBIN" : 2048,
                     "PARFILE" : parfile,
                     "OFFSET0" : 0.0,
                     "OFFSET1" : 0.0,
                     "OFFSET2" : 0.0,
                     "OFFSET3" : 0.0,
                     "SCALE0" : 1.0,
                     "SCALE1" : 1.0,
                     "SCALE2" : 1.0,
                     "SCALE3" : 1.0,
                     "NBITSREQ" : 8,
                     "STT_IMJD" : 66380,
                     "STT_SMJD" : 10010,
                     "STT_OFFS" : 0,
                     "LST" : 0,
                     "BMAJ" : 0.0,
                     "BMIN": 0.0,
#                     "SCANNUM": scannum,
#                     "TBIN" : tbin,
#                     "CHAN_BW" : chbw,
                     "RA" : 0.1,
                     "DEC" : 0.1,
                     "AZ" : 0.0,
                     "ZA": 0.0,
                     "FFTLEN" : 262144,
                     "OVERLAP" : 0,#98304,
                     "BLOCSIZE" : 134217728,
                     "CHAN_DM" : par.DM,
                     }
    
#    specific_params ={
#                     "DATAPORT" : 61000, 
#                     "DATAHOST" : "10.17.0.220",
#                     "OBSFREQ" : 1380.0,
#                     "OBSBW" : -25.0, 
#                     "OBSNCHAN" : 512/8,
#                     "DATADIR" : "/data/gpu/partial/gpu09",
                     
#                     }

    return common_params    
    #g9.setParams(**common_params)
    #g9.setParams(**specific_params)
    
def softSync(tvg=False,noNet=False):
    return sync(tvg=tvg,noNet=noNet,softPps=True)

def sync(tvg=False,noNet=False,softPps=False, at=None):
    
    if tvg:
        tvgm = 0x08
    else:
        tvgm = 0x00
    if noNet:
        tvgm += 0x30
    stopData()
    time.sleep(0.3)
    r1.write_int('pol0_ddcout_ctrl',0x8)
    r1.write_int('pol0_ddcout_ctrl',0x9)
    r1.write_int('pol0_fftout_ctrl',0x8)
    r1.write_int('pol0_fftout_ctrl',0x9)
    r1.write_int('pol1_ddcout_ctrl',0x8)
    r1.write_int('pol1_ddcout_ctrl',0x9)
    r1.write_int('pol1_fftout_ctrl',0x8)
    r1.write_int('pol1_fftout_ctrl',0x9)
    r1.write_int('dout_ctrl',0x8)
    r1.write_int('dout_ctrl',0x9)
    r1.write_int('ctrl',0 | tvgm)
    if at:
        if time.time() > at:
            raise Exception("Missed arm time! now: %s goal: %s" % (time.ctime(),time.ctime(at)))
        while time.time() < at:
            time.sleep(0.05)
    r1.write_int('ctrl',2 | tvgm)
    print "armed at:",time.ctime()," ",time.time()
    r1.write_int('ctrl',0 | tvgm)
    if softPps:
        r1.write_int('ctrl',1 | tvgm)
    else:
        pass
    time.sleep(1)
    getStatus()
    
def doutSnap(tvg=False):
    r1.write_int('dout_ctrl',0)
    r1.write_int('dout_ctrl',1)
    softSync(tvg=tvg)
    tic = time.time()
    while time.time() - tic < 2:
        if r1.read_uint('dout_status') != 0x80000000:
            break
    else:
        print "Error! snap didnt finish!"
    return np.fromstring(r1.read('dout_bram',1024*8),dtype='>i4')
        