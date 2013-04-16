OPT64 = /home/pulsar64
CFLAGS = -O3 -Wall -DFOLD_USE_INTRINSICS -I$(OPT64)/include
PROGS = sdfits_read machinefits_read
OBJS  = vegas_error.o machinefits_read_cmd.o\
	vegas_time.o \
	median.o \
	hget.o hput.o sla.o vegas_params.o
LIBS = -L$(OPT64)/lib -lcfitsio -L/home/pulsar64/presto/lib -lsla -lm -lpthread
all: $(PROGS)
clean:
	rm -f $(PROGS) *~ *.o
INSTALL_DIR = ../bin
install: $(PROGS)
	mkdir -p $(INSTALL_DIR) && \
	cp -f $(PROGS) $(INSTALL_DIR)
.SECONDEXPANSION:
$(PROGS): $$@.c $(OBJS)
	$(CC) $(CFLAGS) $< -o $@ $(OBJS) $(LIBS)
