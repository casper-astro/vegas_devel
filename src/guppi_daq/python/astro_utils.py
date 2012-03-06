import time
from pyslalib import slalib as s
import numpy as Num
from psr_constants import *

def current_MJD():
    """
    current_MJD():
        Return the current MJD accurate to ~1 sec.
    """
    YY, MM, DD, hh, mm, ss, wday, yday, isdst = time.gmtime()
    mjd_f, J = s.sla_dtf2d(hh, mm, ss)
    mjd_i, J = s.sla_cldj(YY, MM, DD)
    return  mjd_i + mjd_f
    
def azza_to_radec(az, za, MJD, fctr=350.0, atm=1010.0, temp=283.0, humid=0.5, scope='GBT'):
    """
    azza_to_radec(az, za, MJD):
        Return RA and DEC (J2000 in deg) from AZ and ZA (in deg) at MJD.  Keyword params
           are fctr=350.0, atm=1010.0, temp=283.0, humid=0.5, scope='GBT'.
    """
    scope, x, lon, lat, hgt = s.sla_obs(0, scope)
    microns = 3e8/(fctr*1e6)*1e6
    app_rarad, app_decrad =  s.sla_oap('a',az*DEGTORAD,za*DEGTORAD,MJD,
                                       0.0,-lon,lat,hgt,0.0,0.0,temp,atm,humid,microns,0.0065)
    ra2000, dec2000 = s.sla_amp(app_rarad, app_decrad, MJD, 2000.0)
    return ra2000*RADTODEG, dec2000*RADTODEG
                                
def radec_to_azza(ra, dec, MJD, fctr=350.0, atm=1010.0, temp=283.0, humid=0.5, scope='GBT'):
    """
    redec_to_azza(ra, dec, MJD):
        Return AZ and ZA (in deg) from RA and DEC (J2000 in deg) at MJD.  Keyword params
           are fctr=350.0, atm=1010.0, temp=283.0, humid=0.5, scope='GBT'.
    """
    scope, x, lon, lat, hgt = s.sla_obs(0, scope)
    microns = 3e8/(fctr*1e6)*1e6
    app_rarad, app_decrad = s.sla_map(ra*DEGTORAD,dec*DEGTORAD,0.0,0.0,0.0,0.0,2000.0,MJD)
    az, za, hob, rob, dob  = s.sla_aop(app_rarad,app_decrad,MJD,
                                       0.0,-lon,lat,hgt,0.0,0.0,temp,atm,humid,microns,0.0065)
    az = s.sla_dranrm(az)
    return az*RADTODEG, za*RADTODEG

def hms_to_rad(hour, min, sec):
    """
    hms_to_rad(hour, min, sec):
    Convert hours, minutes, and seconds of arc to radians
    """
    if (hour < 0.0): sign = -1
    else: sign = 1
    return sign * SECTORAD * \
           (60.0 * (60.0 * Num.fabs(hour) +
                    Num.fabs(min)) + Num.fabs(sec))

def beam_halfwidth(obs_freq, dish_diam):
    """
    beam_halfwidth(obs_freq, dish_diam):
    Return the telescope beam halfwidth in arcmin
    'obs_freq' = the observing frqeuency in MHz
    'dish_diam' = the telescope diameter in m
    """
    return 1.2*SOL/(obs_freq*10.0**6)/dish_diam*RADTODEG*60/2

if __name__ == "__main__":
    MJD = 54556.290613425925 # Ter5 Coords
    ra, dec = 267.02070, -24.77920
    az, za = radec_to_azza(ra, dec, MJD)
    print ra, dec, az, za, 132.388, 80.578 # These are from a Ter5 obs
    ra, dec = azza_to_radec(az, za, MJD)
    print ra, dec, az, za, 132.388, 80.578  
