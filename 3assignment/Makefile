CC=gcc
CFLAGS=-fopenmp -O2 -lm

.PHONY: sequential
sequential: gray_scott.c main.c
	$(CC) -DGRID_SIZE=$(GRID_SIZE) $(CFLAGS) -o gray_scott gray_scott.c main.c

