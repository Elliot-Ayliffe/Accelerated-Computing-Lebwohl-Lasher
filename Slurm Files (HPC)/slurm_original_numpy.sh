#!/bin/bash

#SBATCH --job-name=PP1_hl
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=00:10:00
#SBATCH --account=PHYS033186
#SBATCH --partition=teach_cpu

# Load Python module 
module load languages/python/3.12.3

# Run Python Script setting arguments
python LebwohlLasher.py 50 50 0.5 0

# Obtain the profiling text file (sorted by internal time)
python -m cProfile -s time LebwohlLasher.py 50 50 0.5 0 > prof.txt 
