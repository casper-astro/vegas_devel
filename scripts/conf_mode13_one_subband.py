import corr, logging, pylab, time, struct, sys
from numpy import *

katcp_port = 7147
timeout = 10

roach = 'roach03'
dest_ip = 10*(2**24)+145
source_ip = 10*(2**24)+4
dest_port = 60000

#tx_core_name = 'ten_GbE'

mac_base = (2 << 40) + (2<<32)
fabric_port = 60000

boffile='mode13_one_subband_2011_Aug_23_0039.bof'

def exit_fail():
    print 'FAILURE DETECTED. Log entries:\n',lh.printMessages()
    try:
        fpga.stop()
    except: pass
    raise
    exit()

def exit_clean():
    try:
        fpga.stop()
    except: pass
    exit()

def offset_waveform(n_cycles, offset):
    """Create a sin+cos waveform occupying the upper and lower 16 bits, respectively.
    n_cycles is an integer and refers to the number of cycles.
    offset is an integer from 0-7 corresponding to one of the eight time series.
    With a 150 Mhz clock, the resulting waveform has a frequency of
    f_lo = 150 Mhz / 512 * n_cycles."""
    time = arange(512)*8 + offset
    radians = time/4096.*2.*pi
    sinwave = int16(sin(n_cycles*radians)*32767) #32767 = 2**15 - 1
    coswave = int16(cos(n_cycles*radians)*32767)
    waveform = int32(sinwave)*2**16 + coswave #lower 16 bits cosine, upper 16 bits sine
    #pylab.plot(time, waveform, 'o')
    return waveform

def pack_subband(subband, n_cycles):
    for lo in range(8):
        device_name = 'subband'+str(subband)+'_lo_'+str(lo)
        waveform = offset_waveform(n_cycles, lo)
        packwave = struct.pack('>512l', *waveform)
        fpga.write(device_name, packwave)

try:
    lh = corr.log_handlers.DebugLogHandler()
    logger = logging.getLogger(roach)
    logger.addHandler(lh)
    logger.setLevel(10)

    print('Connecting to server %s... '%(roach)),
    fpga = corr.katcp_wrapper.FpgaClient(roach, katcp_port, timeout)
    time.sleep(1)

    if fpga.is_connected():
        print 'ok\n'
    else:
        print 'ERROR connecting to server %s.\n'%(roach)
        exit_fail()

    print '------------------------'
    print 'Programming FPGA...',
    fpga.progdev('') #Unprogram the device
    time.sleep(1)
    sys.stdout.flush()
    fpga.progdev(boffile)
    print 'ok'

    print '---------------------------'
    print 'Configuring transmitter core...',
    time.sleep(2)
    sys.stdout.flush()
    fpga.tap_start('tap0','ten_GbE',mac_base+source_ip,source_ip,fabric_port)
    time.sleep(2)
    print 'done'
    
    print '---------------------------'
    print 'Setting-up destination addresses...',
    sys.stdout.flush()
    fpga.write_int('dest_ip',dest_ip)
    fpga.write_int('port_select',1)
    fpga.write_int('dest_port',fabric_port)
    fpga.write_int('port_select',1)
    print 'done'

    print '---------------------------'
    print 'Resetting cores and counters...',
    sys.stdout.flush()
    fpga.write_int('eth_rst',1)
    fpga.write_int('eth_rst',0)
    fpga.write_int('cnt_rst', 1)
    fpga.write_int('cnt_rst', 0)
    print 'done'

    print '---------------------------'
    print 'Writing Lookup-Tables to Registers'
    pack_subband(1, 8) #Mixing with 2343750 Hz 
    print 'done'

except KeyboardInterrupt:
    exit_clean()
except:
    exit_fail()
