#!/bin/bash
#SBATCH -p "development" 
#SBATCH -N 1
#SBATCH -t 2:00:00

cd "$(dirname "$1")"
echo "Running SKIRT on $(basename -- "$1") from $(pwd)/ ..."
skirt $(basename -- "$1")
