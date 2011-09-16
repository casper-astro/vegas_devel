#!/bin/bash
# Set environment variables for GUPPI, bash version
# Lines that have been commented out are lines that were either modified
# or removed by Simon Scott, to allow installation on otto.
echo "This script is specific to spec-hpc-xx"
echo "Setting GUPPI, GUPPI_DIR, CUDA, PYSLALIB, VEGAS_INCL/BIN/LIB, PATH, PYTHONPATH and LD_LIBRARY_PATH for VEGAS..."

# User must set this variable (GUPPI) for their home directory
# No other variables should need changing
export GUPPI=~/workspace/gbt_devel/src

export GUPPI_DIR=$GUPPI/guppi_daq

export CUDA=/opt/vegas/cuda

export PYSLALIB=/opt/vegas/lib/python2.5/site-packages/pyslalib

export VEGAS_INCL=/opt/vegas/include
export VEGAS_BIN=/opt/vegas/bin
export VEGAS_LIB=/opt/vegas/lib

export PATH=$GUPPI_DIR/bin:$CUDA/bin:$VEGAS_BIN:$PATH

export PYTHONPATH=$GUPPI/lib/python/site-packages:$GUPPI/lib/python:$GUPPI_DIR/python:/opt/vegas/lib/python2.5/site-packages:$PYTHONPATH

export LD_LIBRARY_PATH=$PYSLALIB:$VEGAS_LIB:$CUDA/lib64:$LD_LIBRARY_PATH

