# Makefile for VEGAS

# C compiler and flags
CC = gcc
CFLAGS = -std=gnu99 -pedantic -Wall -O3
# NVCC compiler and flags
NVCC = nvcc
NVFLAGS = --ptxas-options=-v --compiler-bindir=/usr/bin/gcc-4.3 -O3

# linker options
LFLAGS_FFTW3 = -lfftw3f
LFLAGS_CUFFT = -lcufft
LFLAGS_PGPLOT = -lcpgplot
LFLAGS_MATH = -lm

# bin directory
BINDIR = ./bin

all: vegas_cpu_standalone vegas_gpu_standalone vegas_gentestdata tags

# single-sub-band mode, CPU version
vegas_cpu_standalone: vegas_cpu_standalone.c vegas_cpu_standalone.h
	$(CC) $(CFLAGS) vegas_cpu_standalone.c $(LFLAGS_MATH) $(LFLAGS_FFTW3) $(LFLAGS_PGPLOT) -o $(BINDIR)/$@

# n-sub-band modes, GPU version
vegas_gpu_standalone: vegas_gpu_standalone.cu vegas_gpu_standalone.h
	$(NVCC) $(NVFLAGS) vegas_gpu_standalone.cu -lm $(LFLAGS_CUFFT) $(LFLAGS_PGPLOT) -o $(BINDIR)/$@

# test-data-generation program
vegas_gentestdata: vegas_gentestdata.c
	$(CC) $(CFLAGS) vegas_gentestdata.c $(LFLAGS_MATH) -o $(BINDIR)/$@

tags: *.c *.cu *.h
	ctags --language-force=C *.c *.cu *.h

