#!/bin/bash
#SBATCH --job-name=gray_scott_mpi_test
#SBATCH --output=slurm_logs/test_mpi_%j.out
#SBATCH --error=slurm_logs/test_mpi_%j.err
#SBATCH --ntasks=1                 # X MPI processes
#SBATCH --cpus-per-task=1
#SBATCH --time=00:05:00


module purge
module load GCC
module load OpenMPI

mkdir -p slurm_logs

HOST=$(hostname)
GRID_SIZE=256
CORES=$SLURM_NTASKS

# Recompile with correct grid size
echo "Compiling with GRID_SIZE=$GRID_SIZE..."
# make clean
# make GRID_SIZE=$GRID_SIZE mpi
mpicc -O3 -fopenmp -DGRID_SIZE=256 -o gray_scott gray_scott.c -lm

# run and capture stdout
OUTPUT=$(mpirun -np $CORES ./gray_scott)


# parse Mean V and Time
MEAN=$(echo "$OUTPUT" | grep "Mean V" | sed -E 's/.*Mean V = *([0-9]+\.[0-9]+).*/\1/')
TIME=$(echo "$OUTPUT" | grep "Time =" | sed -E 's/.*Time = *([0-9]+\.[0-9]+) sec.*/\1/')

# write CSV
echo "$HOST,MPI,$GRID_SIZE,$CORES,$MEAN,$TIME" | tee -a timings_MPI.csv