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
nvcc -diag-suppress 550 -Xcompiler -fopenmp -O2 -lcuda -lcudart -lmpi -o par_gray_scott parallel_gray_scott.cu main_parallel.c


# Example arguments:
#   grid_size iterations blockSizeX blockSizeY dt du dv f k
ARGS="128 5000 16 16 1.0 0.16 0.08 0.06 0.062"

# Run the simulation
# mpirun -np $SLURM_NTASKS ./par_gray_scott $ARGS

# Parallel timings
for N in 256 512 1024 2048 4096; do
  for BX in 8 16 32; do
    BY=$BX
    ARGS="$N 5000 $BX $BY 1.0 0.16 0.08 0.06 0.062"
    T=$(mpirun -np $SLURM_NTASKS ./par_gray_scott $ARGS | grep -Eo '[0-9]+\.[0-9]+$')
    echo "parallel,$N,$BX,$BY,$T" >> timings_parallel.csv
  done
done
