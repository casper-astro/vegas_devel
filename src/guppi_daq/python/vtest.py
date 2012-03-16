import os
import numpy as np
from matplotlib import pyplot as plt
from corr import katcp_wrapper
import time
import Pyro.naming
import astro_utils # for current_MJD
Pyro.config.PYRO_NS_HOSTNAME='vegas-hpc1.gb.nrao.edu'
ns = Pyro.naming.NameServerLocator().getNS()

# this dictionary maps hpc number to final octet of hpc tenge ip address
tgedict = {1: 32,
           2: 33,
           3: 39,
           4: 37,
           5: 36,
           6: 38,
           7: 34,
           8: 35}
def remoteExec(node,cmd):
    os.system("ssh vegas-hpc%d 'source myvegas.bash; %s'" %  (node,cmd))
def remoteExecAll(cmd):
    for node in range(1,9):
        remoteExec(node,cmd)


if 'vsd' not in globals():
    vsd = {}
for rn in range(1,9):
    rv = 'vs%d' % rn #each VegasServer will be accessed by vs1, vs2, etc
    if rv not in globals():
        try:
            print "connecting to VegasServer",rn
            vs = ns.resolve('VegasServer%d' % rn).getProxy()
            vs._setTimeout(2)
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
def setupNet(rns = range(1,9),starttap=False):
    for rn in rns:
        roach = roachd[rn]
        hostip = subnet + tgedict[rn]
        roachip = subnet + 64 + rn -1
        if starttap:
            roach.tap_start('tap0','gbe0',mac_base + roachip, roachip, fabric_port)
        roach.write_int('dest_ip',hostip)
        roach.write_int('dest_port',60000)
        if vsd.has_key(rn):
            ip = '10.17.0.%d' % (64+rn-1)
            vsd[rn].setParams(DATAHOST=ip,DATAPORT=60000)
            
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
        x[k::8] = a0[k::4]
        x[k+4::8] = a1[k::4]
        y[k::8] = b0[k::4]
        y[k+4::8] = b1[k::4]
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

def sync(rns=range(1,9)):
    print "Preparing"
    for rn in rns:
        roachd[rn].write_int('sg_sync',0x12) #disable pps and sync
        roachd[rn].write_int('arm',0)
    print "Waiting for even second...",
    while np.mod(time.time(),1) > 0.1:
        pass
    while np.mod(time.time(),1) < 0.1:
        pass
    tic = time.time()
    print tic
    print "Arming Roaches"
    for rn in rns:
        roachd[rn].write_int('sg_sync',0x14)
        roachd[rn].write_int('arm',1)
        roachd[rn].write_int('arm',0)
        
    print "Done in", (time.time()-tic)
    print "Setting STTMJD"
    mjd = astro_utils.current_MJD()
    for rn in rns:
        vsd[rn].setParams(STTMJD=mjd)
    print "Done in", (time.time() - tic)
    while (time.time()-tic) <1.2:
        pass
    print "Should have PPS now, disarming roaches"
    for rn in rns:
        roachd[rn].write_int('sg_sync',0x10)
#        roachd[rn].write_int('arm',0)

def adcSummary(rns=range(1,9)):
    
    for rn in rns:
        try:
            x,y = getAdcSnap(roachd[rn])
        except:
            try:
                x,y = getAdcSnap9b(roachd[rn])
            except:
                pass
        xrms = x.std()
        yrms = y.std()
        xdb = 20*np.log10(xrms/20.0)
        ydb = 20*np.log10(yrms/20.0)
        print "ROACH %d : %5.1f  %5.1f dB  ||  %5.1f  %5.1f dB" % (rn,xrms,xdb,yrms,ydb)
def plotAdcs(rns=range(1,9)):
    data = {}
    for rn in rns:
        try:
            data[rn] = getAdcSnap(roachd[rn])
        except:
            try:
                data[rn] = getAdcSnap9b(roachd[rn])
            except:
                pass
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
def programAll(boffile,rns = range(1,9)):
    raw_input("are you sure you want to reprogram?")
    for rn in rns:
        roach = roachd[rn]
        print "programming",rn
        try:
            roach.progdev('')
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


cfs = np.array([18.55,19.8,21.05,22.3,23.55,24.8,18.55,17.0])*1e3 - 775 + 720.0
cfs = cfs*1e6
def setupFakePulsar(nodes=range(1,9),fpgaclk=360e6,frqs=cfs,sideband=-1):
    n = np.arange(8)
    clk = fpgaclk
    if frqs is None:
        frqs = 18e9 - (np.ceil(150e6/(clk*4/1024.0))*clk*4/1024.0) + ((clk*2)*(2*n+1))-((np.ceil(150e6/(clk*4/1024.))*clk*4/1024.0)*n)
    frqd = dict(zip(n+1,frqs))
    esr = fpgaclk*8 # effective sample rate
    
    pfb_rate = sideband*esr/(2*1024.0)
    for node in nodes:
        vsd[node].setParams(EFSAMPFR=esr,
                            NCHAN=1024,
                            EXPOSURE=1e-6,
                            SUB0FREQ=frqd[node],
                            OBSFREQ=frqd[node],
                            CHAN_BW = pfb_rate,
                            FPGACLK = fpgaclk
                            ) #exposure should be ~0 to get every single spectrum
        
    pass
def checkNTP(nodes=range(1,9)):
    tset = {}
    for node in nodes:
        tset[node] = vsd[node].getTime()
    print "Time offsets relative to node:", nodes[0]
    t0 = tset[nodes[0]]
    for node,t in tset.items():
        print "Node",node, " delta : ",(t0-t)
def startHPC(nodes=range(1,9),basedir='/mnt/bdata1'):
    dname = os.path.join(basedir,time.strftime('%Y.%m.%d_%H%M%S',time.gmtime()))
    for node in nodes:
        vsd[node].setParams(DATADIR=dname,FILENUM=0)
        cmd = "ssh vegas-hpc%d 'source myvegas.bash; pkill -SIGINT vegas_hpc_hbw; touch $VEGAS/logs/node%d_hpc_hbw.log; chmod a+rw $VEGAS/logs/node%d_hpc_hbw.log; mkdir %s; chmod a+rwx %s; nohup vegas_hpc_hbw &> $VEGAS/logs/node%d_hpc_hbw.log < /dev/null &'" % (node,node,node,dname,dname,node)
        print cmd
        os.system(cmd)
	
    print "Created data directories:",dname

def updateFromGBT(nodes=range(1,9)):
    for node in nodes:
        vsd[node].updateFromGBT()
def setupShmemDefaults(nodes=range(1,9)):
    for node in nodes:
        vs = vsd[node]
        vs.setParams(PKTFMT='SPEAD',NPOL=2,
                     NCHAN=1024, NSUBBAND=1,
                     SUB0FREQ=1e9,
                     SUB1FREQ=1e9,
                     SUB2FREQ=1e9,
                     SUB3FREQ=1e9,
                     SUB4FREQ=1e9,
                     SUB5FREQ=1e9,
                     SUB6FREQ=1e9,
                     SUB7FREQ=1e9,
                     EXPOSURE=1e-6,
                     FPGACLK=360e6,
                     EFSAMPFR=360e6*8,
                     INSTRUME="VEGAS",
                     ELEV = 0.0,
                     OBJECT = "unknown_obj",
                     FILENUM = 0
                     )
    updateFromGBT(nodes)
                     
def stopHPC(nodes=range(1,9)):
    for node in nodes:
        os.system("ssh vegas-hpc%d 'pkill -SIGINT vegas_hpc_hbw'" % node)


def Reinit(nodes=range(1,9)):
    for k in nodes:
        print "Killing VegasServer, hpc_hbw node:",k
        cmd = "pkill -SIGINT vegas_hpc_hbw; pkill -f VegasServer;"
        remoteExec(k,cmd)
        
        print "ipcclean",k
        cmd = "ipcclean &> $VEGAS/logs/ipcclean.%d.log" % k
        remoteExec(k,cmd)
        
        print "check guppi status",k
        cmd = "$VEGAS_DIR/bin/check_guppi_status &> $VEGAS/logs/check_guppi_status.%d.log" % k
        remoteExec(k,cmd)
        
        print "dbuf1  ",k
        cmd = "$VEGAS_DIR/bin/check_guppi_databuf -c -i1 -n2 -s32 -t1 &> $VEGAS/logs/check_guppi_databuf1.%d.log" % k
        remoteExec(k,cmd)
        print "dbuf2  ",k
        cmd = "$VEGAS_DIR/bin/check_guppi_databuf -c -i2 -n8 -s16 -t2 &> $VEGAS/logs/check_guppi_databuf2.%d.log" % k
        remoteExec(k,cmd)
        print "dbuf3  ",k
        cmd = "$VEGAS_DIR/bin/check_guppi_databuf -c -i3 -n8 -s16 -t2 &> $VEGAS/logs/check_guppi_databuf3.%d.log" % k
        remoteExec(k,cmd)
        
#        print "setup shmem",k
#        cmd = "$VEGAS_DIR/bin/vegas_setup_shmem_mode01"
#        remoteExec(k,cmd)
        
        print "starting vegasserver",k
        cmd = "nohup python2.7 VegasServer.py %d &> $VEGAS/logs/VegasServer%d.log </dev/null &" % (k,k)
        remoteExec(k,cmd)
        print "done with node",k
