from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension, BuildExtension
import os

setup(
    name='naive_gemm_cuda',
    ext_modules=[
        CUDAExtension(
            name='naive_gemm_cuda',
            sources=[
                'naive.cpp',
                'naive_1.cu',
            ],
            extra_compile_args={
                'cxx': ['-O3'],
                'nvcc': [
                    '-O3',
                    '--use_fast_math',
                    '-arch=sm_89',
                ]
            }
        )
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)
