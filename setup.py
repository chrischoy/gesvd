from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension

setup(
    name='gesvd',
    version="0.1",
    ext_modules=[
        CppExtension('gesvd_cpp', ['gesvd.cpp']),
    ],
    scripts=['gesvd.py', '__init__.py'],
    cmdclass={
        'build_ext': BuildExtension
    })
