#!/bin/bash
#SBATCH --job-name=gray_scott_sim
#SBATCH --output=slurm_logs/gray_scott_%j.out
#SBATCH --error=slurm_logs/gray_scott_%j.err
#SBATCH --ntasks=1                   # Total MPI processes
#SBATCH --cpus-per-task=1            # Threads per process (OpenMP)
#SBATCH --nodes=1
#SBATCH --time=01:00:00
#SBATCH --partition=gpu
#SBATCH --exclusive
#SBATCH --nodelist=wn222

module load GCC

HOST="$(hostname)"

# Set OpenMP threads
# export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OMP_NUM_THREADS=1

# echo "Hostname,Version,GridSize,Time" > timings_sequential.csv

# Sequential timings
for N in 256 512 1024 2048 4096; do
    make GRID_SIZE=$N sequential
    OUTPUT=$(./gray_scott)
    TIME=$(echo "$OUTPUT" | grep "Elapsed time" | grep -Eo "[0-9]+\.[0-9]+")
    echo "$HOST,sequential,$N,$TIME" >> timings_sequential.csv
done
