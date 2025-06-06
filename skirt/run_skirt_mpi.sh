#!/bin/bash
#SBATCH -p "development" 
#SBATCH -N 2               # total number of MPI nodes 
#SBATCH -n 8               # total number of MPI tasks (should scale linearly with N given a similar memory requirement; n divided by N is preferably a multiple of 56 and 8 < 56*N/n < 16)
#SBATCH -t 2:00:00

N=$SLURM_JOB_NUM_NODES
n=$SLURM_NTASKS
CORES_PER_NODE=56 # configured for Frontera default CLX nodes, set to 112 for large memory nodes


THREADS_PER_TASK=$((CORES_PER_NODE * N / n))
# Typically, keep (number of threads per task) x (total number of MPI tasks) / (total number of MPI nodes) <= 56 
# This is because Frontera's default CLX nodes have 56 cores and hyperthreading is not enabled on Frontera
# For faster performance/efficiency, reduce THREADS_PER_TASK to as low as possible (>~8), while increasing n and keeping 
# THREADS_PER_TASK x (n/N) = 56 and N constant, until you run into memory issues. Adjust N as desired afterwards.
# Note that the CLX nodes have 192GB shared across the 56 cores, and 144GB /tmp partition local storage on an SSD.
# See https://skirt.ugent.be/root/_user_parallel.html for more information on SKIRT parallelization.
# See https://docs.tacc.utexas.edu/hpc/frontera/ for more information on Frontera's architecture.

echo "Using $N nodes, each running ~$((n / N)) tasks ($n/$N), each running $THREADS_PER_TASK threads (~$CORES_PER_NODE/$((n / N)))." # all numbers will line up if done in nice multiples

module purge
module load cmake/3.24.2
module load intel/19.1.1
module load impi/19.0.7
#module load mvapich2-x/2.3    # alternative MPI library

cd "$(dirname "$1")"

export OMP_NUM_THREADS=THREADS_PER_TASK    # OpenMP threads per MPI rank
export IBRUN_QUIET=1

echo "Running SKIRT on $(basename -- "$1") from $(pwd)/ ..."
mpirun /home1/09737/jbardati/SKIRT/release/SKIRT/main/skirt -t $THREADS_PER_TASK -b $(basename -- "$1")
#ibrun /home1/09737/jbardati/SKIRT/release/SKIRT/main/skirt -t $THREADS_PER_TASK -b $(basename -- "$1")    # gives errors!!

echo "Job ended."
sacct -j $SLURM_JOBID --format=JobID,JobName,Partition,MaxRSS,Elapsed,ExitCode
exit
