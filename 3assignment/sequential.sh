#!/bin/bash
#SBATCH --job-name=gray_scott_sim
#SBATCH --output=slurm_logs/gray_scott_%j.out
#SBATCH --error=slurm_logs/gray_scott_%j.err
#SBATCH --ntasks=1                   # Total MPI processes
#SBATCH --cpus-per-task=4            # Threads per process (OpenMP)
#SBATCH --gres=gpu:1                 
#SBATCH --time=00:30:00              
#SBATCH --partition=gpu              

module load CUDA
module load OpenMPI
module load GCC

# Set OpenMP threads
# export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OMP_NUM_THREADS=1

# Compile (adjust for MPI + OpenMP + CUDA)
nvcc -diag-suppress 550 -Xcompiler -fopenmp -O2 -lcuda -lcudart -lmpi -o gray_scott gray_scott.cu main.c

echo "Version,GridSize,Time" > timings_sequential.csv

# Sequential timings
for N in 256 512 1024 2048 4096; do
    sed -i "s/\.n = [0-9]*/.n = $N/" main.c
    nvcc -diag-suppress 550 -Xcompiler -fopenmp -O2 -lcuda -lcudart -lmpi -o gray_scott gray_scott.cu main.c
    T=$(mpirun -np $SLURM_NTASKS ./gray_scott | grep -Eo '[0-9]+\.[0-9]+$')
    echo "sequential,$N,$T" >> timings_sequential.csv
done
