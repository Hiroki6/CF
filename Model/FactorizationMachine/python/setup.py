from setuptools import setup, find_packages, Extension
from Cython.Distutils import build_ext
import numpy

setup(
    packages=find_packages(),
    cmdclass = {'build_ext': build_ext},
    ext_modules = [Extension("cython_FM", ["cython_FM.pyx"],
    						 include_dirs=[numpy.get_include()])]
)

