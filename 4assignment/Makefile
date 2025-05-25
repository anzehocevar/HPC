CC=gcc
CFLAGS=-fopenmp -O2 -lm
MPICC=mpicc

GRID_SIZE ?= 128

.PHONY: sequential
sequential: gray_scott.c
	$(CC) -DGRID_SIZE=$(GRID_SIZE) $(CFLAGS) -o gray_scott gray_scott.c

.PHONY: parallel
parallel: par_gray_scott.c
	$(MPICC) -DGRID_SIZE=$(GRID_SIZE) $(CFLAGS) -lmpi -o par_gray_scott par_gray_scott.c
