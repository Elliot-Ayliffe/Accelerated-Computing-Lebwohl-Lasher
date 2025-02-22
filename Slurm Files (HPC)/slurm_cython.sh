#!/bin/bash

#SBATCH --job-name=PP1_hl
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8           # Set this to the number of threads you want if you are running parallel cython 
#SBATCH --time=00:10:00
#SBATCH --account=PHYS033186
#SBATCH --partition=teach_cpu

# Load Python module 
module load languages/python/3.12.3

# Load compiler 
module load gcc/12.3.0    

# First compile the code by running the following command on the login node (didn't work through the slurm submission)
#python cython_setup.py build_ext -fi

# Run Cython run script 
python cython_run.py 50 50 0.5 0

# If you are running parallel cython (openmp) add the number of threads on the end of the arguments
python parallel_cython_run.py 50 50 0.5 0 8

