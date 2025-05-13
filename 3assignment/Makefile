.PHONY: sequential
sequential: gray_scott.cu main.c
	nvcc -DGRID_SIZE=$(GRID_SIZE) -diag-suppress 550 -Xcompiler -fopenmp -O2 -lcuda -lcudart -lmpi -o gray_scott gray_scott.cu main.c

