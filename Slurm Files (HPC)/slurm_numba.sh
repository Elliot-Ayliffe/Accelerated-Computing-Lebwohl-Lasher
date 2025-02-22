#!/bin/bash

#SBATCH --job-name=PP1_hl
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8           # Set this to the number of threads you want if you are running parallel numba 
#SBATCH --time=00:10:00
#SBATCH --account=PHYS033186
#SBATCH --partition=teach_cpu

# Load Python module 
module load languages/python/3.12.3


# You must first create a virtual environment and install numba.
# Activate virtual environment containing numba module as BC4 does not have numba module.
# Environment must contain all necessary modules to run the program
source ~/numba_env/bin/activate  

# Run Python Script 
python parallel_numba.py 50 50 0.5 0



# Obtain the profiling text file (sorted by internal time)
#python -m cProfile -s time numba.py 50 50 0.5 0 8 > prof.txt 

deactivate # Deactivate the virtal environment 