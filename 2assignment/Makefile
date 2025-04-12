sequential: sequential.c
	gcc -O2 -lm --openmp sequential.c -o sequential

basic: basic.cu
	nvcc -diag-suppress 550 -O2 -lm basic.cu -o basic

advanced: advanced.cu
	nvcc -diag-suppress 550 -O2 -lm advanced.cu -o advanced

sample: sample.cu
	nvcc -diag-suppress 550 -O2 -lm sample.cu -o sample

