# Makefile to install GUPPI C code
# Commented lines are lines that were modified by Simon Scott to allow installation on otto.

OPT_FLAGS =

ifdef PAYLOAD
	OPT_FLAGS += -DPAYLOAD_SIZE=$(PAYLOAD)
endif

ifdef RAW_DISK
	OPT_FLAGS += -DRAW_DISK
endif

ifdef NULL_DISK
	OPT_FLAGS += -DNULL_DISK
endif

CFLAGS = -g -O3 -Wall -D_GNU_SOURCE -I$(CUDA)/include -I$(VEGAS_INCL) \
		-I$(PYSLALIB) -I$(GUPPI_DIR)/src $(OPT_FLAGS)

NVCCFLAGS = -arch=sm_20 -I$(CUDA)/include -I$(VEGAS_INCL)

LIBS = -L$(PYSLALIB) -L$(VEGAS_LIB) -lcfitsio -lsla -lm -lpthread
CUDA_LIBS = -L$(CUDA)/lib64 -lcufft -lcuda -lcudart -lrt -lm -lpthread

PROGS = check_guppi_databuf check_guppi_status clean_guppi_shmem
OBJS  = guppi_status.o guppi_databuf.o guppi_udp.o guppi_error.o \
	guppi_params.o guppi_time.o guppi_thread_args.o \
	write_sdfits.o misc_utils.o \
	hget.o hput.o sla.o
THREAD_PROGS = test_net_thread vegas_hpc_hbw
THREAD_OBJS  = vegas_net_thread.o vegas_rawdisk_thread.o \
	        guppi_sdfits_thread.o guppi_accum_thread.o \
	        guppi_null_thread.o guppi_fake_net_thread.o
CUDA_OBJS = pfb_gpu.o pfb_gpu_kernels.o vegas_pfb_thread.o

all: $(PROGS) $(THREAD_PROGS) vegas_hpc_lbw 
clean:
	rm -f $(PROGS) $(THREAD_PROGS) *~ *.o sdfits.tgz test_psrfits_0*.fits *.ptx
INSTALL_DIR = ../bin
install: $(PROGS) $(THREAD_PROGS) vegas_hpc_lbw
	mkdir -p $(INSTALL_DIR) && \
	cp -f $(PROGS) $(THREAD_PROGS) vegas_hpc_lbw $(INSTALL_DIR)
sdfits.tgz: sdfits.h read_sdfits.c \
    guppi_SDFITS_template.txt \
	tar cvzf $@ $^
find_dropped_blocks: find_dropped_blocks.o 
#	$(CC) $(CFLAGS) $< -o $@ -L$(OPT64)/lib -lcfitsio -lm
	$(CC) $(CFLAGS) $< -o $@ -lcfitsio -lm
%.o : %.cu
	nvcc -c $(NVCCFLAGS) $< -o $@

vegas_hpc_lbw: vegas_hpc_lbw.c $(THREAD_OBJS) $(OBJS) $(CUDA_OBJS)
	$(CC) $(CFLAGS) $(CUDA_CFLAGS) $< -o $@ $(THREAD_OBJS) \
		$(CUDA_OBJS) $(OBJS) $(LIBS) $(CUDA_LIBS)

.SECONDEXPANSION:
$(PROGS): $$@.c $(OBJS)
	$(CC) $(CFLAGS) $< -o $@ $(OBJS) $(LIBS) $(THREAD_LIBS)
$(THREAD_PROGS): $$@.c $(THREAD_OBJS) $(OBJS)
	$(CC) $(CFLAGS) $< -o $@ $(THREAD_OBJS) $(OBJS) $(LIBS)
