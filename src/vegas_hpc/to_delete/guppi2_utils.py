# guppi2_utils.py
# Some useful funcs for coherent dedispersion setup

def node(idx):
    """
    node(idx):
        Return the hostname for the given processing node
        based on its index number (0-7).
    """
    cfg = open('/opt/64bit/guppi/guppi_daq/gpu_nodes.cfg','r')
    nodelist = []
    for line in cfg:
        line = line.rstrip(' \n')
        nodelist.append(line)
    cfg.close()
    #nodelist = ('gpu1', 'gpu2', 'gpu3', 'gpu4', 
    #        'gpu5', 'gpu6', 'gpu7', 'gpu8')
    return nodelist[idx]

def dm_from_parfile(parfile):
    """
    dm_from_parfile(parfile):
        Read DM value out of a parfile and return it.
    """
    pf = open(parfile, 'r')
    for line in pf:
        fields = line.split()
        key = fields[0]
        val = fields[1]
        if key == 'DM':
            pf.close()
            return float(val)
    pf.close()
    return 0.0

def fft_size_params(rf,bw,nchan,dm,max_databuf_mb=128):
    """
    fft_size_params(rf,bw,nchan,dm,max_databuf_mb=128):
        Returns a tuple of size parameters (fftlen, overlap, blocsize)
        given the input rf (center of band), bw, nchan, 
        DM, and optional max databuf size in MB.
    """
    # Overlap needs to be rounded to a integer number of packets
    # This assumes 8-bit 2-pol data (4 bytes per samp) and 8
    # processing nodes.  Also GPU folding requires fftlen-overlap 
    # to be a multiple of 64.
    # TODO: figure out best overlap for coherent search mode.  For
    # now, make it a multiple of 512
    pkt_size = 8192
    bytes_per_samp = 4
    node_nchan = nchan / 8
    round_fac = pkt_size / bytes_per_samp / node_nchan
    #if (round_fac<64):  round_fac=64
    if (round_fac<512):  round_fac=512
    rf_ghz = (rf - abs(bw)/2.0)/1.0e3
    chan_bw = bw / nchan
    overlap_samp = 8.3 * dm * chan_bw**2 / rf_ghz**3
    overlap_r = round_fac * (int(overlap_samp)/round_fac + 1)
    # Rough FFT length optimization based on GPU testing
    fftlen = 16*1024
    if overlap_r<=1024: fftlen=32*1024
    elif overlap_r<=2048: fftlen=64*1024
    elif overlap_r<=16*1024: fftlen=128*1024
    elif overlap_r<=64*1024: fftlen=256*1024
    while fftlen<2*overlap_r: fftlen *= 2
    # Calculate blocsize to hold an integer number of FFTs
    # Uses same assumptions as above
    max_npts_per_chan = max_databuf_mb*1024*1024/bytes_per_samp/node_nchan
    nfft = (max_npts_per_chan - overlap_r)/(fftlen - overlap_r)
    npts_per_chan = nfft*(fftlen-overlap_r) + overlap_r
    blocsize = int(npts_per_chan*node_nchan*bytes_per_samp)
    return (fftlen, overlap_r, blocsize)

