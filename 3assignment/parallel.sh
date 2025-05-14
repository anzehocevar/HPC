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

HOST="$(hostname)"

# Compile CUDA + OpenMP + MPI
# nvcc -diag-suppress 550 -Xcompiler -fopenmp -O2 -lcuda -lcudart -lmpi -o par_gray_scott parallel_gray_scott.cu main_parallel.c

# Example arguments:
#   grid_size iterations blockSizeX blockSizeY dt du dv f k
ARGS="128 5000 16 16 1.0 0.16 0.08 0.06 0.062"

# Run the simulation
# mpirun -np $SLURM_NTASKS ./par_gray_scott $ARGS
N_PROCS=1

# TYPE="shared_memory"
# TYPE="parallel"
TYPE="advanced"
OUTPUT_CSV="timings_${TYPE}.csv"

echo "Hostname,Version,N,BlockSizeX,BlockSizeY,Time,AvgConcU,AvgConcV" > "$OUTPUT_CSV"

# Parallel timings
for BX in 8 16 24 32; do
  BY=$BX
  make BLOCK_SIZE_X=$BX BLOCK_SIZE_Y=$BY "$TYPE"
  for N in 256 512 1024 2048 4096; do
    ARGS="$N 5000 $BX $BY 1.0 0.16 0.08 0.06 0.062"
    OUTPUT=$(mpirun -np $N_PROCS ./par_gray_scott $ARGS)
    ELAPSED_TIME=$(echo "$OUTPUT" | grep "Elapsed time" | grep -Eo "[0-9]+\.[0-9]+")
    AVG_CONC_U=$(echo "$OUTPUT" | grep "concentration of U" | grep -Eo "[0-9]+\.[0-9]+")
    AVG_CONC_V=$(echo "$OUTPUT" | grep "concentration of V" | grep -Eo "[0-9]+\.[0-9]+")
    echo "$HOST,$TYPE,$N,$BX,$BY,$ELAPSED_TIME,$AVG_CONC_U,$AVG_CONC_V" >> "$OUTPUT_CSV"
  done
done
