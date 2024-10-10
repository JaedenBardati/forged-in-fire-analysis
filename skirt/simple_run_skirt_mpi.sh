#!/bin/bash
#SBATCH -p "development" 
#SBATCH -N 2              # total number of MPI nodes
#SBATCH -n 14              # total number of MPI tasks
#SBATCH -t 0:10:00

module purge
module load cmake/3.24.2
module load intel/19.1.1
module load impi/19.0.7

ibrun /home1/09737/jbardati/SKIRT/release/SKIRT/main/skirt /home1/09737/jbardati/work/skirt/skirt_tests/PanTorus/PanTorus.ski
#ibrun -n 7 /home1/09737/jbardati/SKIRT/release/SKIRT/main/skirt /home1/09737/jbardati/work/skirt/skirt_tests/PanTorus/PanTorus.ski
