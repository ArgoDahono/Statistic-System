import os
import sys

from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext

try:
    import pybind11
    pybind11_include = pybind11.get_include()
except ImportError:
    pybind11_include = os.environ.get(
        'PYBIND11_INCLUDE',
        r'E:\Program Files\pybind11\include' if sys.platform == 'win32' else '/usr/local/include'
    )

try:
    import numpy
    numpy_include = numpy.get_include()
except ImportError:
    numpy_include = os.environ.get(
        'NUMPY_INCLUDE',
        '/usr/local/lib/python3.x/site-packages/numpy/core/include'
    )

if sys.platform == 'win32':
    extra_compile_args = ['/std:c++17', '/O2', '/EHsc']
    extra_link_args = []
else:
    extra_compile_args = ['-std=c++17', '-O3']
    extra_link_args = []

source_files = [
    'bindings.cpp',
    'plasma_leakage.cpp',
]

ext_modules = [
    Extension(
        'plasma_leakage',
        sources=source_files,
        include_dirs=[
            pybind11_include,
            numpy_include,
            os.getcwd(),
        ],
        language='c++',
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args,
    ),
]

class BuildExt(build_ext):
    def build_extensions(self):
        for ext in self.extensions:
            ext.include_dirs.append(pybind11_include)
            ext.include_dirs.append(numpy_include)
            ext.include_dirs.append(os.getcwd())
        super().build_extensions()

setup(
    name='plasma_leakage',
    version='1.0.0',
    author='Research Team',
    description='Plasma Leakage Monte Carlo Simulation Module for Dengue Hemorrhagic Fever Analysis',
    long_description='',
    ext_modules=ext_modules,
    cmdclass={'build_ext': BuildExt},
    zip_safe=False,
    python_requires='>=3.7',
    install_requires=[
        'pybind11>=2.6.0',
        'numpy>=1.19.0',
    ],
)