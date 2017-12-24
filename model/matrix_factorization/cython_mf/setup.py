from distutils.core import setup
from Cython.Build import cythonize
import numpy
from distutils.extension import Extension
from Cython.Distutils import build_ext

Ext_modules = [Extension('cy_mf', ['cy_mf.pyx'], include_dirs=[numpy.get_include()])]

setup(
    name = 'cy_mf app',
    cmdclass = {'build_ext':build_ext},
    ext_modules = Ext_modules
)
