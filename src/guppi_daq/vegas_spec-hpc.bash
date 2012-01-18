#!/bin/bash
# Set environment variables for GUPPI, bash version
# Lines that have been commented out are lines that were either modified
# or removed by Simon Scott, to allow installation on otto.
echo "This script is specific to spec-hpc-xx"
echo "Setting GUPPI, VEGAS_DIR, CUDA, PYSLALIB, VEGAS_INCL/BIN/LIB, PATH, PYTHONPATH and LD_LIBRARY_PATH for VEGAS..."

# Note: user must set the VEGAS variable in their bash startup script

export VEGAS_DIR=$VEGAS/guppi_daq

export CUDA=/opt/local/cuda

export PYSLALIB=/opt/local/lib/python2.5/site-packages/pyslalib

#export VEGAS_INCL=/opt/local/include
#export VEGAS_BIN=/opt/local/bin
export VEGAS_LIB=/home/gbt7/llama64/lib
export VEGAS_LIB_GCC=/usr/lib/gcc/x86_64-redhat-linux/3.4.6

export PATH=$VEGAS_DIR/bin:$CUDA/bin:$VEGAS_BIN:$PATH

export PYTHONPATH=$VEGAS/lib/python/site-packages:$VEGAS/lib/python:$VEGAS_DIR/python:/opt/vegas/lib/python2.5/site-packages:$PYTHONPATH

export LD_LIBRARY_PATH=$PYSLALIB:$VEGAS_LIB:$CUDA/lib64:$LD_LIBRARY_PATH

