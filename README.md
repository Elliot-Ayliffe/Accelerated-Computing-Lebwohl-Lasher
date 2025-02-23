# Lebwohl Lasher Accelerated Computing Project 
SCIFM0004 Accelerated Computing Project: The Lebwohl-Lasher Model of Liquid Crystals.

This project experiments with different acceleration methods (NumPy Vectorisation, Numba, Cython, openMP, MPI) and various method combinations to speed up a Python program that simulates the 2D Lebwohl-Lasher model.

## 1. Navigating the Repository:
Each method/ version of the program has its own folder containing:

- The modified version of the script (e.g. numba_no_vectorisation.py)
- Set up and run scripts (for building and running cython versions)
- **'Outputs'**: Folder containing output files (times and parameter values)
- **'Tests & Validation Checks':** Folder containing plots and statisitcs used to verify that each modified version still works correctly and produces the expected results. These plots and stats are produced by running the script **'Testing_strategy_Script.py'** using the command:
```
python Testing_Strategy_Script.py <original_code_output_file.txt> <new_version_output_file.txt>
```

- Profiling file detailing which functions are doing the most work in the program (e.g. prof_V1_NUmba.txt)

The **'Scaling Tests'** folder investigates how each method/version scales with the size of the model and number of threads (for parallelised versions). It contains:

- **'Scaling_plots.ipynb'**: Jupyter Notebook plotting the data to produce scaling graphs.
- Data files (csv) required for the Jupyter Notebook
- Output files for all runs

**'execution_times_log.xlsx'**: This is a log of all the changes made for each verison and their execution times for various configurations.



## 2. How to Run the Programs:
All programs for this project were ran using HPC on Blue Crystal 4 through a submission of a slurm file. These are provided in the folder **'Slurm Files (HPC)'**

**Command Line Usage**

Original, NumPy Vectorised, and Numba (including parallel) versions:

```
$ python <filename> <ITERATIONS> <LATTICE SIZE> <TEMPERATURE> <PLOTFLAG>
```

Cython (non parallel) versions, first compile the program by running the setup script followed by the run script:

```
$ python <setup script filename> build_ext -fi
```
```
$ python <run script filename> <ITERATIONS> <LATTICE SIZE> <TEMPERATURE> <PLOTFLAG>
```

Parallel Cython runs the setup file in the same way but inlcudes an additional argument for the run script. The number of threads were also specified in the slurm file:

```
$ python <parallel run script> <ITERATIONS> <LATTICE SIZE> <TEMPERATURE> <PLOTFLAG> <NUM THREADS>
```

Parallel Numba versions were run once before timings were recorded to allow cache to store memory. The number of threads were specified in the slurm file.

MPI (mpi4py) versions:

```
$ mpiexec -n <NUM PROCESSES/WORKERS> python <mpi script filename> <ITERATIONS> <LATTICE SIZE> <TEMPERATURE> <PLOTFLAG>
```






