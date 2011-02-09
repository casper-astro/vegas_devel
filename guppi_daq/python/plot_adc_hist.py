#! /usr/bin/env python
from guppi.client import Client
cicada = Client()
import guppi.interpreter

# Check to see if observations are running
do_arm = False
daqstate = cicada.get("DAQ/DAQSTATE")
print "DAQ state = %s" % daqstate
if (daqstate=="stopped"):
    do_arm = True

if (do_arm==False):
    print """
    WARNING:  Observations appear to be currently running, so the HW will
    not be re-armed.  This means data displayed here may not be current.
    """

if (do_arm):
    cicada.arm()

print "\nADC Power level info:"
guppi.interpreter.plot_adc_hist()
