CC=gcc
CFLAGS=-fopenmp -O2 -lm
NVCC=nvcc
NVCFLAGS=-diag-suppress 550 -Xcompiler -fopenmp -O2 -lcuda -lcudart -lmpi

GRID_SIZE ?= 128
BLOCK_SIZE_X ?= 16
BLOCK_SIZE_Y ?= 16

.PHONY: sequential
sequential: gray_scott.c main.c
	$(CC) -DGRID_SIZE=$(GRID_SIZE) $(CFLAGS) -o gray_scott gray_scott.c main.c

.PHONY: parallel
parallel: parallel_gray_scott.cu main_parallel.c
	$(NVCC) -DBLOCK_SIZE_X=$(BLOCK_SIZE_X) -DBLOCK_SIZE_Y=$(BLOCK_SIZE_Y) $(NVCFLAGS) -o par_gray_scott parallel_gray_scott.cu main_parallel.c

.PHONY: shared_memory
shared_memory: parallel_gray_scott.cu main_parallel.c
	$(NVCC) -DUSE_SHARED_MEMORY -DBLOCK_SIZE_X=$(BLOCK_SIZE_X) -DBLOCK_SIZE_Y=$(BLOCK_SIZE_Y) $(NVCFLAGS) -o par_gray_scott parallel_gray_scott.cu main_parallel.c
