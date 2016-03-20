from distutils.core import setup
from Cython.Build import cythonize
from distutils.extension import Extension
from Cython.Distutils import build_ext
import numpy

Ext_modules = [Extension('cython_FM', sources=['cython_FM.pyx', 'cython_FM.c'], include_dirs=[numpy.get_include()])]

setup(
    name        = 'cython_FM app',
    cmdclass    = {'build_ext':build_ext},
    ext_modules = Ext_modules
)
