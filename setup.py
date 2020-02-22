from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension

setup(
    name='gesvd',
    ext_modules=[
        CppExtension('gesvd', ['gesvd.cpp']),
    ],
    cmdclass={
        'build_ext': BuildExtension
    })
