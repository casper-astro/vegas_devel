# Makefile to install GUPPI C code
# Commented lines are lines that were modified by Simon Scott to allow installation on otto.

# OPT64 = /home/pulsar64
# CUDA = /opt/gpu/cuda
CUDA = /usr/local/cuda
# CFLAGS = -g -O3 -Wall -DFOLD_USE_INTRINSICS -I$(OPT64)/include -I$(CUDA)/include
# CFLAGS = -g -O3 -Wall -I$(OPT64)/include -I$(CUDA)/include
CFLAGS = -g -O3 -Wall -D_GNU_SOURCE -I$(CUDA)/include -I$(CFITSIO)/include -I$(PYSLALIB)
# NVCCFLAGS = -g -arch sm_13 -I$(CUDA)/include -I$(OPT64)/include
NVCCFLAGS = -g -arch sm_13 -I$(CUDA)/include -I$(CFITSIO)/include
# NVCCFLAGS = -g -ccbin $(CUDA)/gcc -I$(CUDA)/include -I$(OPT64)/include
PROGS = check_guppi_databuf check_guppi_status clean_guppi_shmem
OBJS  = guppi_status.o guppi_databuf.o guppi_udp.o guppi_error.o \
	guppi_params.o guppi_time.o guppi_thread_args.o \
	write_sdfits.o misc_utils.o \
	hget.o hput.o sla.o
THREAD_PROGS = test_net_thread guppi_daq vegas_hpc_hbw
THREAD_OBJS  = vegas_net_thread.o guppi_rawdisk_thread.o \
	       guppi_sdfits_thread.o guppi_accum_thread.o \
	       guppi_null_thread.o guppi_fake_net_thread.o
CUDA_OBJS = 
# LIBS = -L$(OPT64)/lib -lcfitsio -L$(PRESTO)/lib -lsla -lm -lpthread
LIBS = -L$(CFITSIO)/lib -lcfitsio -L$(PRESTO)/lib -L$(PYSLALIB) -lsla -lm -lpthread
CUDA_LIBS = -L$(CUDA)/lib64 -lcufft -lcuda -lcudart -lrt -lm
all: $(PROGS) $(THREAD_PROGS) guppi_daq vegas_hpc_hbw
clean:
	rm -f $(PROGS) $(THREAD_PROGS) psrfits.tgz *~ *.o test_psrfits_0*.fits *.ptx
INSTALL_DIR = ../bin
install: $(PROGS) $(THREAD_PROGS)
	mkdir -p $(INSTALL_DIR) && \
	cp -f $(PROGS) $(THREAD_PROGS) $(INSTALL_DIR)
psrfits.tgz: sdfits.h read_sdfits.c \
	guppi_PSRFITS_v3.4_fold_template.txt \
	guppi_PSRFITS_v3.4_search_template.txt \
    guppi_SDFITS_template.txt \
	tar cvzf $@ $^
find_dropped_blocks: find_dropped_blocks.o 
#	$(CC) $(CFLAGS) $< -o $@ -L$(OPT64)/lib -lcfitsio -lm
	$(CC) $(CFLAGS) $< -o $@ -lcfitsio -lm
psrfits_subband: 
test_dedisp_thread: 
guppi_daq_dedisp:
guppi_daq_server:
.SECONDEXPANSION:
$(PROGS): $$@.c $(OBJS)
	$(CC) $(CFLAGS) $< -o $@ $(OBJS) $(LIBS) $(THREAD_LIBS)
$(THREAD_PROGS): $$@.c $(THREAD_OBJS) $(OBJS)
	$(CC) $(CFLAGS) $< -o $@ $(THREAD_OBJS) $(OBJS) $(LIBS)
