from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension

setup(
    name='gesvd_cpp',
    ext_modules=[
        CppExtension('gesvd_cpp', ['gesvd.cpp']),
    ],
    cmdclass={
        'build_ext': BuildExtension
    })
