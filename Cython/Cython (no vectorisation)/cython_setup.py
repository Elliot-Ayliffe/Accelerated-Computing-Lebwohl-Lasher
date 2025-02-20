# Cython setup script (no vectorisation)
# Command line: python cython_setup.py build_ext -fi

from setuptools import setup
from Cython.Build import cythonize 
import numpy 

setup(ext_modules=cythonize("cython_no_vectorisation.pyx"), include_dirs=[numpy.get_include()])