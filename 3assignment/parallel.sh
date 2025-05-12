#!/bin/bash
#SBATCH --job-name=parallel_gray_scott_sim
#SBATCH --output=slurm_logs/par_gray_scott_%j.out
#SBATCH --error=slurm_logs/par_gray_scott_%j.err
#SBATCH --ntasks=1                   # Total MPI processes
#SBATCH --cpus-per-task=4           # OpenMP threads per process
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --partition=gpu

module purge
module load GCC
module load CUDA
module load OpenMPI

# Set OpenMP threads
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

# Compile CUDA + OpenMP + MPI
nvcc -Xcompiler -fopenmp -O2 -lcuda -lcudart -lmpi -o par_gray_scott parallel_gray_scott.cu main_parallel.c


# Example arguments:
#   grid_size iterations block_size dt du dv f k
ARGS="256 5000 16 1.0 0.16 0.08 0.06 0.062"

# Run the simulation
# mpirun -np $SLURM_NTASKS ./par_gray_scott $ARGS

echo "mode,grid_size,block_size,init_time,compute_time,avgV" > timings_parallel.csv

# for N in 256 512 1024 2048; do
#   for B in 8 16 32; do
for N in 256; do
  for B in 8; do
    echo "Running: N=$N, B=$B"
    output=$(mpirun -np $SLURM_NTASKS ./par_gray_scott $N 5000 $B 1.0 0.16 0.08 0.06 0.062)
    init_time=$(echo "$output" | grep Init_time | cut -d ':' -f2)
    compute_time=$(echo "$output" | grep Compute_time | cut -d ':' -f2)
    avgV=$(echo "$output" | grep "Average concentration of V" | awk '{print $5}')
    echo "parallel,$N,$B,$init_time,$compute_time,$avgV" >> timings_parallel.csv
  done
done
