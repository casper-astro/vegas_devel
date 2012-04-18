# Makefile to install VEGAS C code
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

ifdef EXT_DISK
	OPT_FLAGS += -DEXT_DISK
endif

CFLAGS = -g -O3 -Wall -D_GNU_SOURCE -I$(CUDA)/include -I$(VEGAS_INCL) \
		-I$(PYSLALIB) -I$(VEGAS_DIR)/src $(OPT_FLAGS)

NVCCFLAGS = -arch=sm_20 -I$(CUDA)/include -I$(VEGAS_INCL)

LIBS = -L. -L$(PYSLALIB) -L$(VEGAS_LIB) -L$(VEGAS_LIB_GCC) -lcfitsio -lsla -lm -lpthread -lg2c
CUDA_LIBS = -L$(CUDA)/lib64 -lcufft -lcuda -lcudart -lrt -lm -lpthread

PROGS = check_vegas_databuf check_vegas_status clean_vegas_shmem
OBJS  = vegas_status.o vegas_databuf.o vegas_udp.o vegas_error.o \
	vegas_params.o vegas_time.o vegas_thread_args.o \
	write_sdfits.o misc_utils.o \
	hget.o hput.o sla.o
THREAD_PROGS = test_net_thread vegas_hpc_hbw
THREAD_OBJS  = vegas_net_thread.o vegas_rawdisk_thread.o \
	        vegas_sdfits_thread.o vegas_accum_thread.o \
	        vegas_null_thread.o vegas_fake_net_thread.o
CUDA_OBJS = pfb_gpu.o pfb_gpu_kernels.o vegas_pfb_thread.o

all: $(PROGS) $(THREAD_PROGS) vegas_hpc_lbw 
clean:
	rm -f $(PROGS) $(THREAD_PROGS) *~ *.o sdfits.tgz test_psrfits_0*.fits *.ptx
INSTALL_DIR = ../bin
install: $(PROGS) $(THREAD_PROGS) vegas_hpc_lbw
	mkdir -p $(INSTALL_DIR) && \
	cp -f $(PROGS) $(THREAD_PROGS) vegas_hpc_lbw $(INSTALL_DIR)
sdfits.tgz: sdfits.h read_sdfits.c \
    vegas_SDFITS_template.txt \
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
