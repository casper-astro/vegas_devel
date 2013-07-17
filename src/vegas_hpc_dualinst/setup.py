#!/usr/bin/env python

"""VEGAS HPC package

Data acquisition Python package for VEGAS.
"""

from distutils.core import setup, Extension
import os
import sys

srcdir = 'python'
doclines = __doc__.split("\n")

setup(
    name        = 'vegas_hpc'
  , version     = '0.1'
  , packages    = ['vegas_hpc']
  , package_dir = {'vegas_hpc' : srcdir}
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
