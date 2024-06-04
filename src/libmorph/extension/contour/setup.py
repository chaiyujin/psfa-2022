import os

from pybind11.setup_helpers import Pybind11Extension
from setuptools import setup

_dir = os.path.dirname(os.path.abspath(__file__))
ext_modules = [
    Pybind11Extension("contour_finder", [os.path.join(_dir, "pybind.cpp")]),
]

setup(ext_modules=ext_modules)
