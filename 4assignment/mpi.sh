#!/bin/bash
#SBATCH --job-name=gray_scott_mpi_test
#SBATCH --output=slurm_logs/test_mpi_%j.out
#SBATCH --error=slurm_logs/test_mpi_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=64
#SBATCH --cpus-per-task=1
#SBATCH --time=04:00:00
#SBATCH --partition=gpu
#SBATCH --nodelist=gwn02
#SBATCH --exclusive

module purge
module load GCC
module load OpenMPI

mkdir -p slurm_logs

HOST=$(hostname)

# Recompile
# make clean
# TYPE="par_gray_scott"
TYPE="advanced"
make "$TYPE"

# echo "Hostname,Type,GridSize,Cores,Time" > timings_MPI.csv
for grid_size in 256 512 1024 2048 4096; do
    for ncpus in 64 16 4 1; do
        output=$(mpirun -np $ncpus "./${TYPE}" "$grid_size")
        mean=$(echo "$output" | grep "Mean V" | sed -E 's/.*Mean V = *([0-9]+\.[0-9]+).*/\1/')
        time=$(echo "$output" | grep "Time =" | sed -E 's/.*Time = *([0-9]+\.[0-9]+) sec.*/\1/')
        echo "$HOST,$TYPE,$grid_size,$ncpus,$time" >> timings_MPI.csv
        sleep 1
    done
done

