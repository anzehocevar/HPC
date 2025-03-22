#!/bin/bash

#SBATCH --reservation=fri
#SBATCH --job-name=job
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --output=out.txt
#SBATCH --nodelist=wn141
#SBATCH --threads-per-core=1
#SBATCH --exclusive

export OMP_PLACES=cores
export OMP_PROC_BIND=TRUE
# export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

gcc -O2 -lm --openmp SeamCarving.c -o SeamCarving

srun SeqBench.sh
