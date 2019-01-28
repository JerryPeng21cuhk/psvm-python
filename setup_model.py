from distutils.core import setup, Extension
from Cython.Build import cythonize
import numpy


#setup(
#        ext_modules=[
#            Extension("model", ["model.c"],
#                include_dirs=[numpy.get_include()]),
#            ],
#        )

setup(ext_modules=cythonize('cmodel.pyx'),
        include_dirs=[numpy.get_include()])
