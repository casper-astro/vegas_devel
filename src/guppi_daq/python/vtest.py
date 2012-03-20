"""
vtest.py - convenience routines for running Vegas mode 1 testing at Green Bank
--------

*NOTE* Reinit() expects that VegasServer.py is in your home directory and that you have passwordless
ssh access to all HPC nodes setup. Also assumes there is a myvegas.bash script in your home directory that should be
executed to set up the vegas environment (python in particular)

I've been using this by starting ipython --pylab and then
run -i vtest.py
to provide all functions conveniently from the ipython prompt.

Typical setup would be:
If pyro nameserver is not running on vegas-hpc1, it needs to be started
Then do

Reinit()

to setup all shared memory, all Vegas Servers

At this point, quit ipython and start again
ipython --pylab
run -i vest.py
Now you should be all set up with vs1...8
so you can do:
vs1.getSynthFreq()
etc. See VegasServer.py for functions available on vs1,vs2, etc. Unfortunately these are remote method
calls using Pyro, so you can't tab complete.

Now Set up everything:

setupShmemDefaults() #basic shared memory parameters

setSynths(1440) #set all valon synthesiers

programAll('the_latest_boffile.bof') #program the roaches

checkClocks() #make sure all clocks are locked

plotAdcs() # have a look at ADC plots. If any ROACHes are glitching, reprogram them with:
            # programAll('the_latest_boffile.bof',[1,3,4]) #This will reprogram roaches 1,3 and 4
            
setupNet() # setup networking stuff on all roaches

reset() # setup acc_len etc on roaches (they should now be dumping data)

setupFakePulsar() # setup astronomical parameters

startHPC() # start vegas_hpc code

[vs.getParam('NETSTAT') for vs in vsd.values()] # Check that all nodes are 'recieving' (not 'exiting' or 'waiting')

sync() # arm roaches so they start on next PPS. HPC nodes should start takign data

[vs.getParam('NPKT') for vs in vsd.values()] # Check that all HPC are getting packets as expected

[vs.getParam('FILENUM') for vs in vsd.values()] # Watch how many files the HPC code has written

stopHPC() # stop writing data so we can look at it.
"""
import os
import numpy as np
from matplotlib import pyplot as plt
from corr import katcp_wrapper
import time
import Pyro.naming
import astro_utils # for current_MJD
Pyro.config.PYRO_NS_HOSTNAME='vegas-hpc1.gb.nrao.edu'  # This is the machine on which the pyro nameserver is running
ns = Pyro.naming.NameServerLocator().getNS()

# this dictionary maps hpc number to final octet of hpc tenge ip address
# addresses are 10.17.0.x
tgedict = {1: 32,
           2: 33,
           3: 39,
           4: 37,
           5: 36,
           6: 38,
           7: 34,
           8: 35,
           9: 10}
           
def remoteExec(node,cmd):
    """
    Execute system command on remote node using ssh
    
    node : int 1-8 for vegas-hpc1...vegas-hpc8
           or str for arbitrary node
    cmd : str
        cmd to be executed
    """
    if type(node) is int:
        node = 'vegas-hpc%d' % node
    os.system("ssh %s 'source myvegas.bash; %s'" %  (node,cmd))
    
def remoteExecAll(cmd,nodes=range(1,9)):
    """
    Execute system command on specified HPC nodes. Not used much these days
    """
    for node in nodes:
        remoteExec(node,cmd)

#######################################################
# The following sets up the global namespace with access to VegasServers
# and roaches
# vsd is a dictionary of all VegasServers
# vs1, vs2, vs3, etc are the VegasServers
# vses is a list of all VegasServers (not generally used)
#
# roachd is a dictionary of all roaches
# r1, r2, r3,.. are the FpgaClients for each roach
# roaches is a list of all Roaches (not generally used)
#
# You can do things like:
# r1.listbof()
# [vs.getParam('NETSTAT') for vs in vsd.values()]  # returns a list of values of NETSTAT from shared memory, useful to see if HPC code is running and recieveign data
# vs1.setSynthFreq(1440) # set synthesier frequency

if 'vsd' not in globals():
    vsd = {}
for rn in range(1,10):
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
for rn in range(1,10):
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

################### End setting up global namespace

def setSynths(freq,rns = range(1,9)):
    """
    Set Valon synthesizer frequency connected to HPC node
    
    freq : float
        frequency in MHz
    rns : list of ints
        Node numbers (1-8 for vega-hpc, 9 for tofu)
    """
    for rn in rns:
        try:
            vsd[rn].setSynthFreq(freq)
            print "set ",rn
        except Exception, e:
            print "failed to set synth",rn
            print e
            
subnet = 10*2**24 + 17*2**16 #10.17.0.0
mac_base = (2 << 40) + (2<<32)
fabric_port = 60000
def setupNet(rns = range(1,9),starttap=False):
    """
    Configure networking for roach and associated HPC.
    
    Sets up dest_ip and dest_port registers on ROACH and DATAHOST and DATAPORT parameters in the status shared memory.
    Optionally starts tap on roaches
    
    rns : list of ints
        Node numbers (1-8 for vega-hpc, 9 for tofu)
    starttap : bool
        should the starttap be run? No need to set this to False, as you can starttap twice with no ill effects.

    """
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
    """
    Reset roach and perform basic setup (including acc_len and period) Data should start flowing if setupNet has been done

    rns : list of ints
        Node numbers (1-8 for vega-hpc, 9 for tofu)    
    acc_len : int
        Actual accumulation length in spectra. Note: 1 will be subtracted this number to be written to the acc_len register
    """
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
    """
    get ADC snapshot from ROACHes with mainline high speed bof files
    
    roach : FpgaClient (r1...r9)
    
    Returns:
        x,y tuple of numpy arrays with adc values
        
    Expects a global trigger register called 'raw_snap_trig' and individual 32-bit snap blocks called adc[01]_snap[01]
    """
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

def getAdcSnapTest(roach):
    """
    get ADC snapshot from ROACHes with an adc test boffile
    
    roach : FpgaClient (r1...r9)
    
    Returns:
        x,y tuple of numpy arrays with adc values
        
    Expects a global trigger register called 'trig' and individual 64-bit snap blocks called adcsnap_[ab]
    """

    roach.write_int('trig',0)
    for ctrl in [('adcsnap_%s_ctrl' % (x,)) for x in ['a','b']]:
        roach.write_int(ctrl,0)
        roach.write_int(ctrl,5)
    roach.write_int('trig',1)
    a0 = np.fromstring(roach.read('adcsnap_a_bram_msb',8192),dtype='int8')
    a1 = np.fromstring(roach.read('adcsnap_a_bram_lsb',8192),dtype='int8')
    b0 = np.fromstring(roach.read('adcsnap_b_bram_msb',8192),dtype='int8')
    b1 = np.fromstring(roach.read('adcsnap_b_bram_lsb',8192),dtype='int8')
    x = np.empty((16384,),dtype='float')
    y = np.empty((16384,),dtype='float')    
    for k in range(4):
        x[k::8] = a0[k::4]
        x[k+4::8] = a1[k::4]
        y[k::8] = b0[k::4]
        y[k+4::8] = b1[k::4]
    return x,y


def getAdcSnap9b(roach):
    """
    get ADC snapshot from ROACHes with old 9b boffiles
    
    roach : FpgaClient (r1...r9)
    
    Returns:
        x,y tuple of numpy arrays with adc values
        
    Expects single 128-bit snap with common ctrl register
    """
    
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
    """
    Reset roaches on PPS
    
    Waits for appropriate fraction of second to arm roaches. Sets STTMJD status shared memory parameter on HPC hosts
    
    rns : list of ints
        Node numbers (1-8 for vega-hpc, 9 for tofu)    
    """
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
    """
    print summary of ADC levels for setting attenuators
    
    rns : list of ints
        Node numbers (1-8 for vega-hpc, 9 for tofu)    
    """
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
        
def doAdcTest(bof,clks=[800,1200,1440,1500],basedir='/home/sandboxes/vegastest/data/adctests',note='',rns=range(1,9)):
    """
    Test routine for testing phase offset of ADC clock on data integrity
    """
    for clk in clks:
        fbase = os.path.join(basedir,('%s_%dMHz%s' % (bof,clk,note)))
        print fbase
        setSynths(clk,rns=rns)
        programAll(bof,force=True,rns=rns)
        fh = open(fbase + '_clockReport.txt','w')
        for rn in rns:
            try:
                fclk = roachd[rn].est_brd_clk()
                if np.abs(clk/4.0 - fclk) > 3:
                    print "Warning: Roach %d clock not locked: %.2f" % (rn,fclk)
            except:
                fclk = -1
            fh.write('Roach %d: %.2f MHz\n' % (rn,fclk))
        fh.close()
        print "wrote clock report"
        plotAdcs(rns=rns,fn= fbase)
        plt.close('all')
        
        
def plotAdcs(rns=range(1,9),fn = None):
    """
    Prepare summary plots of ADC snapshots
    
    Produces plots of the histogram, power spectral density, and a snippet of raw data for each ADC from each roach
    
    rns : list of ints
        Node numbers (1-8 for vega-hpc) NOTE: Does not work for node 9, tofu currently
    fn : string (optional)
        Start of file location to save plots. _hist.png etc will be appended to form final filenames
    """
    data = {}
    for rn in rns:
        try:
            data[rn] = getAdcSnap(roachd[rn])
        except:
            try:
                data[rn] = getAdcSnap9b(roachd[rn])
            except:
                try:
                    data[rn] = getAdcSnapTest(roachd[rn])
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
            tax.plot(x[:300],'b',lw=1.5)  
            tax.plot(y[:300],'r',lw=1.5)
            tax.text(0.05,0.95,("Roach %d" % (rx+1)),size='small',transform=tax.transAxes,ha='left',va='top')
    plt.draw()   
    if fn:
        blah,name = os.path.split(fn)
    else:
        name = ''
    f1.suptitle('Histograms\n' + name,size='small')
    f2.suptitle('Power spectrum\n' + name,size='small')
    f3.suptitle('Time domain\n' + name,size='small')      
    if fn:
        f1.savefig(fn+'_hist.png')
        f2.savefig(fn+'_psd.png')
        f3.savefig(fn+'_time.png')
        
def programAll(boffile,rns = range(1,9),force=True):
    """
    Program many roaches with given boffile
    
    boffile : str
        Name of the boffile
    rns : list of ints
        Node numbers (1-8 for vega-hpc, 9 for tofu)    
    force : bool
        Used to have an interactive check to avoid accidental reprogramming, but this is not generally used now
    """
    if not force:
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

def setupSlaves(rns=range(2,9),mss=0x2):
    """
    Configure given ROACHes as switching signal slaves.
    
    Basically just writes value of mss to the master_slave_sel register on each given roach.
    """
    for rn in rns:
        roachd[rn].write_int('ssg_master_slave_sel',mss)

def execAll(cmd,*args,rns=range(1,9)):
    """
    Execute a function on each ROACH
    
    cmd : str
        Function to execute. I.e. 'listbof', 'write_int', etc
    *args : positional arguments for cmd
    rns : list of ints
        Node numbers (1-8 for vega-hpc, 9 for tofu)    
    
    E.g.
    execAll('read_int','acc_len')
    execAll('write_int','acc_len',767,rns=[1,3,5])
    """
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
    """
    Check the status of all valons and all roach clocks
    
    Prints the valon frquency in MHz, whether it's locked to 10 MHz, and the roach clock returned by est_brd_clk
    """
    for rn,roach in roachd.items():
    
        print "Valon: ",rn,' ',
        try:
            res = vsd[rn].getSynthFreq()
            print res,'MHz - ',
            res = vsd[rn].getSynthLocked()
            print res,' ',
        except:
            print "<unavailable> ",
        print "Roach",rn,":",
        try:
            print roach.est_brd_clk(),
        except:
            print "<unavailable> ",
        print "MHz"


cfs = np.array([18.55,19.8,21.05,22.3,23.55,24.8,18.55,17.0])*1e3 - 775 + 720.0
cfs = cfs*1e6
def setupFakePulsar(nodes=range(1,9),fpgaclk=360e6,frqs=cfs,sideband=-1):
    """
    Set up astronomical parameters in shared status memory
    pass frqs = None for Andrew's center frequency formula
    """
    n = np.arange(len(nodes))
    clk = fpgaclk
    if frqs is None:
#        frqs = 18e9 - (np.ceil(150e6/(clk*4/1024.0))*clk*4/1024.0) + ((clk*2)*(2*n+1))-((np.ceil(150e6/(clk*4/1024.))*clk*4/1024.0)*n)
        bandtop = 26e9
        fbw = fpgaclk*4
        chanbw = fbw/1024.0
        uic = 1400e6
        lic = 100e6
        frqs = bandtop - fbw*(2*n+1)/2.0 + np.ceil(lic/chanbw)*chanbw*n + np.ceil((fbw-uic)/chanbw)*chanbw*n
    frqd = dict(zip(nodes,frqs))
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
    """
    Check that NTP is working on all nodes
    
    Prints difference in system time between first node and all other nodes
    """
    tset = {}
    for node in nodes:
        tset[node] = vsd[node].getTime()
    print "Time offsets relative to node:", nodes[0]
    t0 = tset[nodes[0]]
    for node,t in tset.items():
        print "Node",node, " delta : ",(t0-t)
        
def startHPC(nodes=range(1,9),basedir='/mnt/bdata1'):
    """
    Start the vegas_hpc_hbw code on hpc nodes
    
    nodes : list of ints
        Node numbers (1-8 for vega-hpc, 9 for tofu)    
    basedir : str
        location in which to save data on each node
        
    A subdirectory of basedir named for the current date and time will be created on each node.
    This directory will be set to the DATADIR status memory value
    Any residual vegas_hpc_hbw programs will be killed
    The vegas_hpc_hbw will be started and output logged to $VEGAS/logs/node#_hpc_hbw.log
    """
    dname = os.path.join(basedir,time.strftime('%Y.%m.%d_%H%M%S',time.gmtime()))
    for node in nodes:
        vsd[node].setParams(DATADIR=dname,FILENUM=0)
        if node == 9:
            nodename = 'tofu'
        else:
            nodename = 'vegas-hpc%d' % node
        cmd = "ssh %s 'source myvegas.bash; pkill -SIGINT vegas_hpc_hbw; touch $VEGAS/logs/node%d_hpc_hbw.log; chmod a+rw $VEGAS/logs/node%d_hpc_hbw.log; mkdir %s; chmod a+rwx %s; nohup vegas_hpc_hbw &> $VEGAS/logs/node%d_hpc_hbw.log < /dev/null &'" % (nodename,node,node,dname,dname,node)
        print cmd
        os.system(cmd)
	
    print "Created data directories:",dname

def updateFromGBT(nodes=range(1,9)):
    """
    Update status memory of all nodes with parameters from GBT status
    """
    for node in nodes:
        vsd[node].updateFromGBT()
def setupShmemDefaults(nodes=range(1,9)):
    """
    Run this after Reinit to make sure all nodes have default values
    """
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
    """
    Stop vegas_hpc_hbw on all nodes. Tries to use SIGINT so that partial files get neatly closed.
    """
    for node in nodes:
        if node == 9:
            nodename = 'tofu'
        else:
            nodename = 'vegas-hpc%d' % node
        os.system("ssh %s 'pkill -SIGINT vegas_hpc_hbw'" % nodename)


def Reinit(nodes=range(1,9)):
    """
    Bring down software infrastructure and bring it back up again.
    
    * Kills any vegas_hpc_hbw and VegasServers
    * does ipcclean to remove shared memory
    * initializes shared status memory with check_guppi_status
    * initializes data buffers with check_guppi_databuf
    * starts VegasServer
    
    NOTE: After running this, you'll want to quit ipython and start again, then rerun this vtest.py file
    This will regather the VegasServers since the Pyro proxies will be stale after killing and restarting 
    the VegasServers
    """
    for k in nodes:
        print "Killing VegasServer, hpc_hbw node:",k
        cmd = "pkill -SIGINT vegas_hpc_hbw; pkill -f VegasServer;"
        print cmd
        remoteExec(k,cmd)
        
        print "ipcclean",k
        cmd = "ipcclean &> $VEGAS/logs/ipcclean.%s.log" % str(k)
        print  cmd
        remoteExec(k,cmd)
        
        print "check guppi status",k
        cmd = "$VEGAS_DIR/bin/check_guppi_status &> $VEGAS/logs/check_guppi_status.%s.log" % str(k)
        print cmd
        remoteExec(k,cmd)
        
        print "dbuf1  ",k
        cmd = "$VEGAS_DIR/bin/check_guppi_databuf -c -i1 -n2 -s32 -t1 &> $VEGAS/logs/check_guppi_databuf1.%s.log" % str(k)
        print cmd
        remoteExec(k,cmd)
        print "dbuf2  ",k
        cmd = "$VEGAS_DIR/bin/check_guppi_databuf -c -i2 -n8 -s16 -t2 &> $VEGAS/logs/check_guppi_databuf2.%s.log" % str(k)
        print cmd
        remoteExec(k,cmd)
        print "dbuf3  ",k
        cmd = "$VEGAS_DIR/bin/check_guppi_databuf -c -i3 -n8 -s16 -t2 &> $VEGAS/logs/check_guppi_databuf3.%s.log" % str(k)
        print cmd
        remoteExec(k,cmd)
        
#        print "setup shmem",k
#        cmd = "$VEGAS_DIR/bin/vegas_setup_shmem_mode01"
#        remoteExec(k,cmd)
        
        print "starting vegasserver",k
        if k == 'tofu':
            nodenum = 9
        else:
            nodenum = k
        cmd = "nohup python2.7 VegasServer.py %d &> $VEGAS/logs/VegasServer%d.log </dev/null &" % (nodenum,nodenum)
        print cmd
        remoteExec(k,cmd)
        print "done with node",k
