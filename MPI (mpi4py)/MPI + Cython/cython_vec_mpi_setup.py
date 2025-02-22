# Cython setup script (with numpy vectorisation & MPI)
# Command line: python cython_numpy_vec_setup.py build_ext -fi

from setuptools import setup
from Cython.Build import cythonize 
import numpy 

setup(ext_modules=cythonize("cython_vec_mpi.pyx"), include_dirs=[numpy.get_include()])