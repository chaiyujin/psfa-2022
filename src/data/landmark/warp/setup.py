# from distutils.core import setup
import numpy
from Cython.Build import cythonize
from setuptools import Extension, setup

setup(
    ext_modules=cythonize(
        Extension(
            "mask_utils",
            sources=["mask_utils.pyx"],
            include_dirs=[numpy.get_include()],
        ),
        language_level="3",
    )
)
