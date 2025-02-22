# Cython setup script (parallelised)
# Command line: python parallel_cython_setup.py build_ext -fi

from setuptools import setup, Extension
from Cython.Build import cythonize 
import numpy 


ext_modules = [Extension("parallel_cython_openmp",
                         ["parallel_cython_openmp.pyx"],
                         extra_compile_args=["-fopenmp"],
                         extra_link_args=["-fopenmp"],
                         )]
setup(ext_modules=cythonize(ext_modules), include_dirs=[numpy.get_include()])