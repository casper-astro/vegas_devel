#!/usr/bin/env python

"""GUPPI DAQ package

Data acquisition Python package for the Green Bank Ultimate Pulsar Processing
Instrument (GUPPI).
"""

from distutils.core import setup, Extension
import os
import sys

srcdir = 'python'
doclines = __doc__.split("\n")

setup(
    name        = 'guppi_daq'
  , version     = '0.1'
  , packages    = ['guppi_daq']
  , package_dir = {'guppi_daq' : srcdir}
  , maintainer = "NRAO"
  , ext_modules=[Extension('_possem',
                           [os.path.join(srcdir, 'possem.i')])]
  # , maintainer_email = ""
  # , url = ""
  , license = "http://www.gnu.org/copyleft/gpl.html"
  , platforms = ["any"]
  , description = doclines[0]
  , long_description = "\n".join(doclines[2:])
  )
