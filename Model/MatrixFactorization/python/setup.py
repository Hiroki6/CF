from distutils.core import setup
from Cython.Build import cythonize
import numpy
from distutils.extension import Extension
from Cython.Distutils import build_ext

setup(
    cmdclass = {'build_ext':build_ext},
    ext_modules = cythonize("cythonMF.pyx"),
    include_dirs = [numpy.get_include()]
)
