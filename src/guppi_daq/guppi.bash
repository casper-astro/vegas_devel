#!/bin/bash
# Set environment variables for GUPPI, bash version
# Lines that have been commented out are lines that were either modified
# or removed by Simon Scott, to allow installation on otto.
echo "This script is specific to CASPER (otto)"
echo "Setting GUPPI_DIR, PATH, PYTHONPATH, LD_LIBRARY_PATH, TEMPO, PRESTO and PGPLOT_DIR for GUPPI..."

# User must set this variable (GUPPI) for their home directory
# No other variables should need changing
export GUPPI=~/guppi

# export GUPPI=$OPT64/guppi
export GUPPI_DIR=$GUPPI/guppi_daq

# Simon added these two lines line
export CFITSIO=/opt/cfitsio	
export PYSLALIB=/opt/pyslalib	

# export PRESTO=$PSR64/presto
export PRESTO=/opt/presto

# export PATH=$PSR64/bin:$PRESTO/bin:$GUPPI_DIR/bin:$GUPPI/bin:$OPT64/bin:$PATH
export PATH=$PRESTO/bin:$GUPPI_DIR/bin:/usr/local/cuda/bin:$PATH

#export PYTHONPATH=$PSR64/lib/python:$PSR64/lib/python/site-packages:$PRESTO/lib/python:$GUPPI/lib/python/site-packages:$GUPPI/lib/python:$GUPPI_DIR/python:$PYTHONPATH
export PYTHONPATH=$PRESTO/lib/python:$GUPPI/lib/python/site-packages:$GUPPI/lib/python:$GUPPI_DIR/python:$PYTHONPATH

# export LD_LIBRARY_PATH=$PSR64/lib:$OPT64/lib:$PGPLOT_DIR:$PRESTO/lib
export LD_LIBRARY_PATH=$PRESTO/lib:$CFITSIO/lib:$PYSLALIB

# export HEADAS=/home/pulsar64/src/heasoft-6.6.2/x86_64-unknown-linux-gnu-libc2.3.4
# alias ftools=". $HEADAS/headas-init.sh"
# PSR64=/home/pulsar64
# OPT64=/opt/64bit
# export TEMPO=$PSR64/tempo
# export PGPLOT_DIR=$PSR64/pgplot


