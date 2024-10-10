#!/bin/bash
#SBATCH -p "development" 
#SBATCH -N 1               # total number of MPI nodes
#SBATCH -n 1               # total number of MPI tasks
#SBATCH -t 2:00:00

NTHREADSPERTASK=56
# Typically, keep (number of threads per task) x (total number of MPI tasks) / (total number of MPI nodes) <= 56 
# This is because Frontera's CLX nodes (default) have 56 cores and hyperthreading is not enabled on Frontera
# For faster performance/efficiency, reduce NTHREADSPERTASK to as low as possible (>~8), while increasing n and keeping 
# NTHREADSPERTASK x (n/N) = 56 and N constant, until you run into memory issues. Adjust N as desired afterwards.
# Note that the CLX nodes have 192GB shared across the 56 cores, and 144GB /tmp partition local storage on an SSD.
# See https://skirt.ugent.be/root/_user_parallel.html for more information on SKIRT parallelization.
# See https://docs.tacc.utexas.edu/hpc/frontera/ for more information on Frontera's architecture.

module purge
module load cmake/3.24.2
module load intel/19.1.1
module load impi/19.0.7
#module load mvapich2-x/2.3    # alternative MPI library

cd "$(dirname "$1")"

export OMP_NUM_THREADS=NTHREADSPERTASK    # OpenMP threads per MPI rank
export IBRUN_QUIET=1

echo "Running SKIRT on $(basename -- "$1") from $(pwd)/ ..."
ibrun /home1/09737/jbardati/SKIRT/release/SKIRT/main/skirt -t $NTHREADSPERTASK -b $(basename -- "$1")

echo "Job ended."
sacct -j $SLURM_JOBID --format=JobID,JobName,Partition,MaxRSS,Elapsed,ExitCode
exit