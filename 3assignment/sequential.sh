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
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

# Compile once
nvcc -Xcompiler -fopenmp -O2 -lcuda -lcudart -lmpi -o gray_scott gray_scott.cu main.c

echo "mode,grid_size,block_size,init_time,compute_time,avgV" > timings_sequential.csv

# 512 1024 2048 4096
for N in 256; do
    echo "Running N=$N"
    output=$(mpirun -np $SLURM_NTASKS ./gray_scott $N)

    init_time=$(echo "$output" | grep Init_time | cut -d ':' -f2)
    compute_time=$(echo "$output" | grep Compute_time | cut -d ':' -f2)
    avgV=$(echo "$output" | grep "Average concentration of V" | awk '{print $5}')

    echo "sequential,$N,0,$init_time,$compute_time,$avgV" >> timings_sequential.csv
done