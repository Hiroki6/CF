from distutils.core import setup
from Cython.Build import cythonize
from Cython.Distutils import build_ext
import numpy

setup(
    ext_modules = cythonize("cython_FM.pyx"),
    include_dirs = [numpy.get_include()]
)
