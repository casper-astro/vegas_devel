#!/bin/bash
# Set environment variables for GUPPI, bash version
# Lines that have been commented out are lines that were either modified
# or removed by Simon Scott, to allow installation on otto.
echo "This script is specific to otto"
echo "Setting GUPPI, GUPPI_DIR, CUDA, PYSLALIB, VEGAS_INCL/BIN/LIB, PATH, PYTHONPATH and LD_LIBRARY_PATH for VEGAS..."

# User must set this variable (GUPPI) for their home directory
# No other variables should need changing
export GUPPI=~/casper/gbt_devel/src

export GUPPI_DIR=$GUPPI/guppi_daq

# Simon added these lines
export CUDA=/usr/local/cuda
export PYSLALIB=/opt/src/pyslalib	
export CFITSIO=/opt/src/cfitsio	

export VEGAS_INCL=$CFITSIO/include
export VEGAS_BIN=
export VEGAS_LIB=$CFITSIO/lib

# export PATH=$PSR64/bin:$PRESTO/bin:$GUPPI_DIR/bin:$GUPPI/bin:$OPT64/bin:$PATH
export PATH=$GUPPI_DIR/bin:$CUDA/bin:$VEGAS_BIN:$PATH

#export PYTHONPATH=$PSR64/lib/python:$PSR64/lib/python/site-packages:$PRESTO/lib/python:$GUPPI/lib/python/site-packages:$GUPPI/lib/python:$GUPPI_DIR/python:$PYTHONPATH
export PYTHONPATH=$GUPPI/lib/python/site-packages:$GUPPI/lib/python:$GUPPI_DIR/python:$PYTHONPATH

# export LD_LIBRARY_PATH=$PSR64/lib:$OPT64/lib:$PGPLOT_DIR:$PRESTO/lib
export LD_LIBRARY_PATH=$PYSLALIB:$VEGAS_LIB:$CUDA/lib64:$LD_LIBRARY_PATH

# export HEADAS=/home/pulsar64/src/heasoft-6.6.2/x86_64-unknown-linux-gnu-libc2.3.4
# alias ftools=". $HEADAS/headas-init.sh"
# PSR64=/home/pulsar64
# OPT64=/opt/64bit
# export TEMPO=$PSR64/tempo
# export PGPLOT_DIR=$PSR64/pgplot


