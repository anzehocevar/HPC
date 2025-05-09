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

# Compile (adjust for MPI + OpenMP + CUDA)
nvcc -Xcompiler -fopenmp -lcuda -lcudart -lmpi -o gray_scott gray_scott.cu main.c

# Run the simulation with MPI
mpirun -np $SLURM_NTASKS ./gray_scott