# Makefile for VEGAS

# C compiler and flags
CC = gcc
CFLAGS = -std=gnu99 -pedantic -Wall -O3
NVCC = nvcc
NVFLAGS = --compiler-bindir=/usr/bin/gcc-4.3 -O3
LFLAGS_FFTW3 = -lfftw3f
LFLAGS_CUFFT = -lcufft
LFLAGS_PGPLOT = -lcpgplot
LFLAGS_MATH = -lm
BINDIR = ./bin

all: vegasp3mcpu vegasp3mcpu.byte vegasp3mgpu vegasp3mgpu.byte vegasp3mgpu.byte.2dtex vegasp3mgpu.opt vegasp2mgpu.byte tags

# priority 3 mode CPU with converted float input
vegasp3mcpu: vegasp3mcpu.c vegasp3mcpu.h
	$(CC) $(CFLAGS) vegasp3mcpu.c $(LFLAGS_MATH) $(LFLAGS_FFTW3) $(LFLAGS_PGPLOT) -o $(BINDIR)/$@

# priority 3 mode CPU with byte input
vegasp3mcpu.byte: vegasp3mcpu.byte.c vegasp3mcpu.byte.h
	$(CC) $(CFLAGS) vegasp3mcpu.byte.c -lm $(LFLAGS_FFTW3) $(LFLAGS_PGPLOT) -o $(BINDIR)/$@

# priority 3 mode GPU with converted float input
vegasp3mgpu: vegasp3mgpu.cu vegasp3mgpu.h
	$(NVCC) $(NVFLAGS) vegasp3mgpu.cu -lm $(LFLAGS_CUFFT) $(LFLAGS_PGPLOT) -o $(BINDIR)/$@

# priority 3 mode GPU with byte input
vegasp3mgpu.byte: vegasp3mgpu.byte.cu vegasp3mgpu.byte.h
	$(NVCC) $(NVFLAGS) vegasp3mgpu.byte.cu -lm $(LFLAGS_CUFFT) $(LFLAGS_PGPLOT) -o $(BINDIR)/$@

# priority 3 mode GPU with byte input, using 2D textures
vegasp3mgpu.byte.2dtex: vegasp3mgpu.byte.2dtex.cu vegasp3mgpu.byte.2dtex.h
	$(NVCC) $(NVFLAGS) vegasp3mgpu.byte.2dtex.cu -lm $(LFLAGS_CUFFT) $(LFLAGS_PGPLOT) -o $(BINDIR)/$@

# priority 3 mode GPU with byte input, cleaned-up and optimized
vegasp3mgpu.opt: vegasp3mgpu.opt.cu vegasp3mgpu.opt.h
	$(NVCC) $(NVFLAGS) vegasp3mgpu.opt.cu -lm $(LFLAGS_CUFFT) $(LFLAGS_PGPLOT) -o $(BINDIR)/$@

# priority 2 mode GPU with byte input
#vegasp2mgpu.byte: vegasp2mgpu.byte.cu vegasp2mgpu.byte.h
#	$(NVCC) $(NVFLAGS) vegasp2mgpu.byte.cu -lm $(LFLAGS_CUFFT) $(LFLAGS_PGPLOT) -o $(BINDIR)/$@

tags: *.c *.cu *.h
	ctags --language-force=C *.c *.cu *.h

