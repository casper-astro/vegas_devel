OPT64 = /home/pulsar64
CUDA = /opt/gpu/cuda
#CFLAGS = -g -O3 -Wall -DFOLD_USE_INTRINSICS -I$(OPT64)/include -I$(CUDA)/include
CFLAGS = -g -O3 -Wall -I$(OPT64)/include -I$(CUDA)/include
NVCCFLAGS = -g -arch sm_13 -I$(CUDA)/include -I$(OPT64)/include
#NVCCFLAGS = -g -ccbin $(CUDA)/gcc -I$(CUDA)/include -I$(OPT64)/include
PROGS = check_guppi_databuf check_guppi_status clean_guppi_shmem \
	test_udp_recv test_psrfits test_psrfits_read \
	fix_psrfits_polyco
OBJS  = guppi_status.o guppi_databuf.o guppi_udp.o guppi_error.o \
	guppi_params.o guppi_time.o guppi_thread_args.o \
	write_psrfits.o read_psrfits.o misc_utils.o \
	fold.o polyco.o hget.o hput.o sla.o downsample.o \
	dedisperse_utils.o
THREAD_PROGS = test_net_thread guppi_daq guppi_daq_fold 
THREAD_OBJS  = guppi_net_thread.o guppi_rawdisk_thread.o \
	       guppi_psrfits_thread.o guppi_fold_thread.o \
	       guppi_null_thread.o
CUDA_OBJS = guppi_dedisp_thread.o guppi_dedisp_ds_thread.o \
	    fold_gpu.o downsample_gpu.o dedisperse_gpu.o
LIBS = -L$(OPT64)/lib -lcfitsio -L$(PRESTO)/lib -lsla -lm -lpthread
CUDA_LIBS = -L$(CUDA)/lib64 -lcufft -lcuda -lcudart -lrt -lm
all: $(PROGS) $(THREAD_PROGS) guppi_daq psrfits_subband \
	test_dedisp_thread guppi_daq_dedisp guppi_daq_server
clean:
	rm -f $(PROGS) $(THREAD_PROGS) psrfits.tgz *~ *.o test_psrfits_0*.fits *.ptx
INSTALL_DIR = ../bin
install: $(PROGS) $(THREAD_PROGS) guppi_daq psrfits_subband test_dedisp_thread guppi_daq_dedisp guppi_daq_server
	mkdir -p $(INSTALL_DIR) && \
	cp -f $(PROGS) $(THREAD_PROGS) psrfits_subband test_dedisp_thread guppi_daq_dedisp guppi_daq_server $(INSTALL_DIR)
psrfits.tgz: psrfits.h read_psrfits.c write_psrfits.c polyco.c polyco.h \
	guppi_PSRFITS_v3.4_fold_template.txt \
	guppi_PSRFITS_v3.4_search_template.txt
	tar cvzf $@ $^
find_dropped_blocks: find_dropped_blocks.o 
	$(CC) $(CFLAGS) $< -o $@ -L$(OPT64)/lib -lcfitsio -lm
psrfits_subband: psrfits_subband.c psrfits_subband_cmd.o $(OBJS)
	$(CC) $(CFLAGS) $< -o $@ psrfits_subband_cmd.o $(OBJS) $(LIBS)
%.o : %.cu
	nvcc -c $(NVCCFLAGS) $< -o $@

test_dedisp_thread: test_dedisp_thread.c $(THREAD_OBJS) $(OBJS) $(CUDA_OBJS)
	$(CC) $(CFLAGS) $(CUDA_CFLAGS) $< -o $@ $(THREAD_OBJS) \
		$(CUDA_OBJS) $(OBJS) $(LIBS) $(CUDA_LIBS)
guppi_daq_dedisp: guppi_daq_dedisp.c $(THREAD_OBJS) $(OBJS) $(CUDA_OBJS)
	$(CC) $(CFLAGS) $(CUDA_CFLAGS) $< -o $@ $(THREAD_OBJS) \
		$(CUDA_OBJS) $(OBJS) $(LIBS) $(CUDA_LIBS)
guppi_daq_server: guppi_daq_server.c $(THREAD_OBJS) $(OBJS) $(CUDA_OBJS)
	$(CC) $(CFLAGS) $(CUDA_CFLAGS) $< -o $@ $(THREAD_OBJS) \
		$(CUDA_OBJS) $(OBJS) $(LIBS) $(CUDA_LIBS)
.SECONDEXPANSION:
$(PROGS): $$@.c $(OBJS)
	$(CC) $(CFLAGS) $< -o $@ $(OBJS) $(LIBS) $(THREAD_LIBS)
$(THREAD_PROGS): $$@.c $(THREAD_OBJS) $(OBJS)
	$(CC) $(CFLAGS) $< -o $@ $(THREAD_OBJS) $(OBJS) $(LIBS)
