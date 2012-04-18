# Set environment variables for GUPPI, csh version
echo "This script is specific to Green Bank."
echo "Setting GUPPI, PATH, PYTHONPATH, LD_LIBRARY_PATH, TEMPO, PRESTO and PGPLOT_DIR for GUPPI..."

setenv HEADAS /home/pulsar64/src/heasoft-6.6.2/x86_64-unknown-linux-gnu-libc2.3.4
alias ftools "source $HEADAS/headas-init.csh"

set PSR64=/home/pulsar64
set OPT64=/opt/64bit

setenv GUPPI $OPT64/guppi
setenv GUPPI_DIR $GUPPI/guppi_daq
setenv PRESTO $PSR64/presto
setenv PATH $PSR64/bin:$PRESTO/bin:$GUPPI_DIR/bin:$GUPPI/bin:$OPT64/bin:$PATH
setenv PYTHONPATH $PSR64/lib/python:$PSR64/lib/python/site-packages:$PRESTO/lib/python:$GUPPI/lib/python/site-packages:$GUPPI/lib/python:$GUPPI_DIR/python
setenv PGPLOT_DIR $PSR64/pgplot
setenv LD_LIBRARY_PATH $PSR64/lib:$OPT64/lib:${PGPLOT_DIR}:$PRESTO/lib
setenv TEMPO $PSR64/tempo
