#!/bin/bash

#SBATCH --job-name=PP1_hl
#SBATCH --nodes=1
#SBATCH --ntasks=8                  # set this for the number of processes/workers you want 
#SBATCH --time=00:10:00
#SBATCH --account=PHYS033186
#SBATCH --partition=teach_cpu

# Load Python module 
module load languages/python/3.12.3


# Load other modules 
module load gcc/12.3.0    


# Run Python Script 
mpiexec -n 8 python mpi4py_original.py 50 50 0.5 0


