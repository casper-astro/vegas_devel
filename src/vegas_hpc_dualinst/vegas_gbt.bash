#!/bin/bash
# Set environment variables for VEGAS, bash version
# Lines that have been commented out are lines that were either modified
# or removed by Simon Scott, to allow installation on otto.
echo "This script is specific to spec-hpc-xx"
echo "Setting VEGAS, VEGAS_DIR, CUDA, PYSLALIB, VEGAS_INCL/BIN/LIB, PATH, PYTHONPATH and LD_LIBRARY_PATH for VEGAS..."


export MCPYTHON_ROOT=/home/gbt7/newt
export SWIG_VERS=2.0.4
export PYTHON_VERS=2.7.2

export PYRO_NS_HOSTNAME=vegas-hpc1.gb.nrao.edu

# deap
export DEAP=$MCPYTHON_ROOT/extern/deap

# swig
export SWIG_LIB=$MCPYTHON_ROOT/share/swig/2.0.4

# build settings
export INCLUDE=/opt/local/include:$MCPYTHON_ROOT/include:$MCPYTHON_ROOT/lib/python2.7/site-packages/numpy/core/include:$MCPYTHON_ROOT/lib/python2.7/site-packages/numpy/numarray/include:$INCLUDE
export LD_LIBRARY_PATH=$MCPYTHON_ROOT/lib:/opt/local/lib:$LD_LIBRARY_PATH
export MANPATH=$MCPYTHON_ROOT/man:$MANPATH
export PATH=$MCPYTHON_ROOT/bin:$PATH
export PYTHONPATH=$MCPYTHON_ROOT/lib/python2.7/site-packages:$PYTHONPATH

# User must set this variable (VEGAS) for their home directory
# No other variables should need changing

export VEGAS_DIR=$VEGAS/vegas_hpc

export CUDA=/opt/local/cuda42

export PYSLALIB=/home/gbt7/newt/lib/python2.7/site-packages/pyslalib

export VEGAS_INCL=$VEGAS_DIR/src
export VEGAS_BIN=$VEGAS_DIR/bin
export VEGAS_LIB=$VEGAS_DIR/lib
export VEGAS_LIB_GCC=/usr/lib/gcc/x86_64-redhat-linux/3.4.6


export PATH=$VEGAS_DIR/bin:$CUDA/bin:$VEGAS_BIN:$PATH

export PYTHONPATH=/home/sandboxes/vegastest/adc_tests:$VEGAS/lib/python2.7/site-packages:$VEGAS_DIR/lib/python:$VEGAS_DIR/python:/users/gjones/lib/python:/users/gjones/lib/python2.7/site-packages:$PYTHONPATH

export LD_LIBRARY_PATH=$PYSLALIB:$VEGAS_LIB:$CUDA/lib64:$LD_LIBRARY_PATH
