CC=gcc
CFLAGS=-fopenmp -O2 -lm
NVCC=nvcc
NVCFLAGS=-diag-suppress 550 -Xcompiler -fopenmp -O2 -lcuda -lcudart -lmpi

GRID_SIZE ?= 128

.PHONY: sequential
sequential: gray_scott.c main.c
	$(CC) -DGRID_SIZE=$(GRID_SIZE) $(CFLAGS) -o gray_scott gray_scott.c main.c
