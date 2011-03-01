import shm_wrapper as shm
from GBTStatus import GBTStatus
import time, pyfits, possem
import numpy as n
#import psr_utils as psr
import astro_utils as astro
import slalib as s

DEGTORAD = 0.017453292519943295769236907684
RADTODEG = 57.29577951308232087679815481410

def header_from_string(str):
    """
    header_from_string(str):
        Convert an input string (which should be the ASCII header from
            a FITS HFU) into an instantiation of a pyfits 'Header' class.
    """
    cl = cardlist_from_string(str)
    return pyfits.Header(cl)

def card_from_string(str):
    """
    card_from_string(str):
        Return a pyfits 'Card' based on the input 'str', which should
            be an 80-character 'line' from a FITS header.
    """
    card = pyfits.Card()
    return card.fromstring(str)

def cardlist_from_string(str):
    """
    cardlist_from_string(str):
        Return a list of pyfits 'Cards' from the input string.
            'str' should be the ASCII from a FITS header.
    """
    cardlist = []
    numcards = len(str)/80
    for ii in range(numcards):
        str_part = str[ii*80:(ii+1)*80]
        if str_part.strip()=="END":
            break
        else:
            cardlist.append(card_from_string(str_part))
    return cardlist


GUPPI_STATUS_KEY = 16783408
GUPPI_STATUS_SEMID = "/guppi_status"

class guppi_status:

    def __init__(self):
        self.stat_buf = shm.SharedMemoryHandle(GUPPI_STATUS_KEY)
        self.sem = possem.sem_open(GUPPI_STATUS_SEMID, possem.O_CREAT, 00644, 1)
        self.hdr = None
        self.gbtstat = None
        self.read()

    def __getitem__(self, key):
        return self.hdr[key]

    def keys(self):
        return [k for k, v in self.hdr.items()]

    def values(self):
        return [v for k, v in self.hdr.items()]

    def items(self):
        return self.hdr.items()

    def lock(self):
        return possem.sem_wait(self.sem)

    def unlock(self):
        return possem.sem_post(self.sem)

    def read(self):
        self.lock()
        self.hdr = header_from_string(self.stat_buf.read())
        self.unlock()

    def write(self):
        self.lock()
        self.stat_buf.write(repr(self.hdr.ascard)+"END"+" "*77)
        self.unlock()

    def update(self, key, value, comment=None):
        self.hdr.update(key, value, comment)

    def show(self):
        for k, v in self.hdr.items():
            print "'%8s' :"%k, v
        print ""

    def update_with_gbtstatus(self):
        if self.gbtstat is None:
            self.gbtstat = GBTStatus()
        self.gbtstat.collectKVPairs()
        g = self.gbtstat.kvPairs
        self.update("TELESCOP", "GBT")
        self.update("OBSERVER", g['observer'])
        self.update("PROJID", g['data_dir'])
        self.update("FRONTEND", g['receiver'])
        self.update("NRCVR", 2) # I think all the GBT receivers have 2...
        if 'inear' in g['rcvr_pol']:
            self.update("FD_POLN", 'LIN')
        else:
            self.update("FD_POLN", 'CIRC')
        freq = float(g['freq'])
        self.update("OBSFREQ", freq)
        self.update("SRC_NAME", g['source'])
        if g['ant_motion']=='Tracking' or g['ant_motion']=='Guiding':
            self.update("TRK_MODE", 'TRACK')
        elif g['ant_motion']=='Stopped':
            self.update("TRK_MODE", 'DRIFT')
        else:
            self.update("TRK_MODE", 'UNKNOWN')
        self.ra = float(g['j2000_major'].split()[0])
        self.ra_str = self.gbtstat.degrees2hms(self.ra)
        self.dec = float(g['j2000_minor'].split()[0])
        self.dec_str = self.gbtstat.degrees2dms(self.dec)
        self.update("RA_STR", self.ra_str)
        self.update("RA", self.ra)
        self.update("DEC_STR", self.dec_str)
        self.update("DEC", self.dec)
        h, m, s = g['lst'].split(":")
        lst = int(round(astro.hms_to_rad(int(h),int(m),float(s))*86400.0/astro.TWOPI))
        self.update("LST", lst)
        self.update("AZ", float(g['az_actual']))
        self.update("ZA", 90.0-float(g['el_actual']))
        beam_deg = 2.0*astro.beam_halfwidth(freq, 100.0)/60.0
        self.update("BMAJ", beam_deg)
        self.update("BMIN", beam_deg)

    def update_azza(self):
        """
        update_azza():
            Update the AZ and ZA based on the current time with the guppi_status instance.
        """
        (iptr, ang, stat) = s.sla_dafin(self['RA_STR'].replace(':', ' '), 1)
        self.update("RA", ang*15.0*RADTODEG)
        (iptr, ang, stat) = s.sla_dafin(self['DEC_STR'].replace(':', ' '), 1)
        self.update("DEC", ang*RADTODEG)
        MJD = astro.current_MJD()
        az, za = astro.radec_to_azza(self['RA'], self['DEC'], MJD, scope='GBT')
        self.update("AZ", az)
        self.update("ZA", za)

GUPPI_DATABUF_KEY = 12987498

class guppi_databuf:

    def __init__(self,databuf_id=1):
        self.buf = shm.SharedMemoryHandle(GUPPI_DATABUF_KEY+databuf_id-1)
        self.data_type = self.buf.read(NumberOfBytes=64, offset=0)
        packed = self.buf.read(NumberOfBytes=3*8+3*4, offset=64)
        self.struct_size, self.block_size, self.header_size = \
                n.fromstring(packed[0:24], dtype=n.int64)
        self.shmid, self.semid, self.n_block= \
                n.fromstring(packed[24:36], dtype=n.int32)
        self.header_offset = self.struct_size 
        self.data_offset = self.struct_size + self.n_block*self.header_size
        self.dtype = n.int8
        self.read_size = self.block_size
        self.read_all_hdr()

    def read_hdr(self,block):
        if (block<0 or block>=self.n_block):
            raise IndexError, "block %d out of range (n_block=%d)" \
                    % (block, self.n_block)
        self.hdr[block] = header_from_string(self.buf.read(self.header_size,\
                self.header_offset + block*self.header_size))

    def read_all_hdr(self):
        self.hdr = []
        for i in range(self.n_block):
            self.hdr.append(header_from_string(self.buf.read(self.header_size,\
                self.header_offset + i*self.header_size)))

    def data(self,block):
        if (block<0 or block>=self.n_block):
            raise IndexError, "block %d out of range (n_block=%d)" \
                    % (block, self.n_block)
        self.read_hdr(block) 
        try:
            if (self.hdr[block]["OBS_MODE"] == "PSR"):
                self.dtype = n.float
            else:
                self.dype = n.int8
        except KeyError:
            self.dtype = n.int8
        raw = n.fromstring(self.buf.read(self.block_size, \
                self.data_offset + block*self.block_size), \
                dtype=self.dtype)
        try:
            npol = self.hdr[block]["NPOL"]
            nchan = self.hdr[block]["OBSNCHAN"]
            if (self.hdr[block]["OBS_MODE"] == "PSR"):
                #nbin = self.hdr[block]["NBIN"]
                #raw.shape = (nbin, npol, nchan)
                return raw
            else:
                nspec = self.block_size / (npol*nchan)
                raw.shape = (nspec, npol, nchan)
        except KeyError:
            pass
        return raw
        
if __name__=="__main__":
    g = guppi_status()
    g.show()

    print 
    print 'keys:', g.keys()
    print 
    print 'values:', g.values()
    print 
    print 'items:', g.items()
    print 

    g.update_with_gbtstatus()
    g.write()
    g.show()
