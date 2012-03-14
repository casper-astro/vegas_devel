import numpy as np
from matplotlib import pyplot as plt
from corr import katcp_wrapper
import time
import Pyro.naming
Pyro.config.PYRO_NS_HOSTNAME='vegas-hpc1.gb.nrao.edu'
ns = Pyro.naming.NameServerLocator().getNS()
if 'vsd' not in globals():
    vsd = {}
for rn in range(1,9):
    rv = 'vs%d' % rn #each VegasServer will be accessed by vs1, vs2, etc
    if rv not in globals():
        try:
            print "connecting to VegasServer",rn
            vs = ns.resolve('VegasServer%d' % rn).getProxy()
            vs._setTimeout(1)
            globals()[rv] = vs
            vsd[rn] = vs
        except Exception, e:
            print "could not connect to VegasServer",rn
            print e
vses = vsd.values()

if 'roachd' not in globals():
    roachd = {}
for rn in range(1,9):
    rv = 'r%d' % rn #each roach will be accessed by r1, r2, etc
    if rv not in globals():
        try:
            print "connecting to roach",rn
            roach = katcp_wrapper.FpgaClient('vegas-r%d' % rn)
            tic = time.time()
            while time.time() - tic < 2:
                if roach.is_connected():
                    break
                time.sleep(0.1)
            else:
                raise Exception("timeout waiting for roach")
            globals()[rv] = roach
            roachd[rn] = roach
        except Exception, e:
            print "could not connect to roach",rn
            print e
roaches = roachd.values()

def setSynths(freq):
    for vsn,vs in vsd.items():
        try:
            vs.setSynthFreq(freq)
            print "set ",vsn
        except Exception, e:
            print "failed to set synth",vsn
            print e
            
subnet = 10*2**24 + 17*2**16 #10.17.0.0
mac_base = (2 << 40) + (2<<32)
fabric_port = 60000
def startTap(rns = range(1,9)):
    for rn in rns:
        roach = roachd[rn]
        hostip = subnet + 32 + rn -1
        roachip = subnet + 64 + rn -1
        roach.tap_start('tap0','gbe0',mac_base + roachip, roachip, fabric_port)
        roach.write_int('dest_ip',hostip)
        roach.write_int('dest_port',60000)
        if vsd.has_key(rn):
            ip = '10.17.0.%d' % (64+rn-1)
            vsd[rn].setParams(DATAHOST=ip)
def reset(rns = range(1,9),acc_len=768):
    for rn in rns:
        roach = roachd[rn]
        roach.write_int('acc_len',acc_len-1)
        roach.write_int('sg_period',acc_len*32*256-2)
        roach.write_int('sg_sync',0x12)
        roach.write_int('arm',0)
        roach.write_int('arm',1)
        roach.write_int('arm',0)
        
        roach.write_int('sg_sync',0x11)
        
def startHPC(node):
    os.system("blah &> /home/sandboxes/vegastest/logs/node%d" % node)
    


def getAdcSnap(roach):
    roach.write_int('raw_snap_trig',0)
    for ctrl in [('adc%d_snap%d_ctrl' % (x,y)) for x in [0,1] for y in [0,1]]:
        roach.write_int(ctrl,0)
        roach.write_int(ctrl,5)
    roach.write_int('raw_snap_trig',1)
    a0 = np.fromstring(roach.read('adc0_snap0_bram',8192),dtype='int8')
    a1 = np.fromstring(roach.read('adc0_snap1_bram',8192),dtype='int8')
    b0 = np.fromstring(roach.read('adc1_snap0_bram',8192),dtype='int8')
    b1 = np.fromstring(roach.read('adc1_snap1_bram',8192),dtype='int8')
    x = np.empty((16384,),dtype='float')
    y = np.empty((16384,),dtype='float')    
    for k in range(4):
        x[k::8] = a1[k::4]
        x[k+4::8] = a0[k::4]
        y[k::8] = b1[k::4]
        y[k+4::8] = b0[k::4]
    return x,y

def getAdcSnap9b(roach):
    roach.write_int('adcsnap_ctrl',0)
    roach.write_int('adcsnap_ctrl',7)    
    a0 = np.fromstring(roach.read('adcsnap_a0',8192),dtype='int8')
    a1 = np.fromstring(roach.read('adcsnap_a1',8192),dtype='int8')
    b0 = np.fromstring(roach.read('adcsnap_b0',8192),dtype='int8')
    b1 = np.fromstring(roach.read('adcsnap_b1',8192),dtype='int8')
    x = np.empty((16384,),dtype='float')
    y = np.empty((16384,),dtype='float')    
    for k in range(4):
        x[k::8] = a1[k::4]
        x[k+4::8] = a0[k::4]
        y[k::8] = b1[k::4]
        y[k+4::8] = b0[k::4]
    return x,y

def plotAdcs(rns=range(1,9)):
    data = {}
    for rn in rns:
        try:
            data[rn] = getAdcSnap(roachd[rn])
        except Exception,e:
            print "failed to get data from roach",rn
            print e
    (f1,haxs) = plt.subplots(nrows=4,ncols=2)
    haxs = haxs.flatten()

    (f2,spaxs) = plt.subplots(nrows=4,ncols=2)
    spaxs = spaxs.flatten()
    (f3,taxs) = plt.subplots(nrows=4,ncols=2)
    taxs = taxs.flatten()
    for rx in range(8):
        if data.has_key(rx+1):
            hax = haxs[rx]
            x,y = data[rx+1]
            h,b = np.histogram(x,bins=np.arange(-128,128))
            label = 'ADC0: ptp=%.0f, rms=%.1f, %.1f dB' % (x.ptp(),x.std(), 20*np.log10(x.std()/20.0))
            hax.plot(b[:-1],np.log10(h),'b',drawstyle='step',lw=1.5,label=label)
            h,b = np.histogram(y,bins=np.arange(-128,128))
            label = 'ADC1: ptp=%.0f, rms=%.1f, %.1f dB' % (y.ptp(),y.std(), 20*np.log10(y.std()/20.0))
            hax.plot(b[:-1],np.log10(h),'r',drawstyle='step',lw=1.5,label=label)
            hax.legend(loc='lower right',prop=dict(size='xx-small'),title=('Roach %d' % (rx+1)))
            hax.set_xlim(-128,128)
            hax.set_ylim(-3,3)
            spax = spaxs[rx]
            spax.psd(x,Fs=3000.0,NFFT=256,lw=1.5,label='ADC0',color='b')
            spax.psd(y,Fs=3000.0,NFFT=256,lw=1.5,label='ADC1',color='r')
            spax.set_ylabel('dB')
            spax.legend(loc='upper right',prop=dict(size='x-small'),title=('Roach %d' % (rx+1))) 
            
            tax = taxs[rx]
            tax.plot(x[:200],'b',lw=1.5)  
            tax.plot(y[:200],'r',lw=1.5)
    plt.draw()   
    f1.suptitle('Histograms')
    f2.suptitle('Power spectrum')
    f3.suptitle('Time domain')      
def programAll(boffile):
    raw_input("are you sure you want to reprogram all roaches?")
    for rn,roach in roachd.items():
        print "programming",rn
        try:
            roach.progdev(boffile)
        except Exception,e:
            print "failed to program roach",rn
            print e
            
def execAll(cmd,*args):
    results = {}
    for rn, roach in roachd.items():
        try:
            fun = getattr(roach,cmd)
            results[rn] = fun(*args)
        except AttributeError, e:
            raise e
        except Exception, e:
            print "failed on roach",rn
            print e
    return results
                
def checkClocks():
    for rn,roach in roachd.items():
    
        print "Valon: ",rn,' ',
        try:
            res = vsd[rn].getSynthFreq()
            print res,'MHz - ',
            res = vsd[rn].getSynthLocked()
            print res,' ',
        except:
            print "<unavailable> ",
        print "Roach",rn,":",roach.est_brd_clk(),"MHz"
