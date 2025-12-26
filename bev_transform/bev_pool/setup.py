"""Setup script for building the BEV Pool CUDA extension.

This builds the bev_pool_ext extension from source using PyTorch's C++/CUDA
extension builder. The extension provides efficient CUDA kernels for BEV pooling
operations used in camera-to-BEV transformations.
"""

import os
import torch
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension, CppExtension


def make_cuda_ext():
    """Create CUDA extension if CUDA is available, otherwise CPU-only extension."""
    
    define_macros = []
    extra_compile_args = {"cxx": []}
    
    sources = [
        "src/bev_pool_cpu.cpp",
    ]
    
    # Check if CUDA is available
    if torch.cuda.is_available() and torch.version.cuda is not None:
        define_macros += [("WITH_CUDA", None)]
        extension = CUDAExtension
        
        # Add CUDA-specific compilation flags
        extra_compile_args["nvcc"] = [
            "-D__CUDA_NO_HALF_OPERATORS__",
            "-D__CUDA_NO_HALF_CONVERSIONS__",
            "-D__CUDA_NO_HALF2_OPERATORS__",
            # Add compute capabilities for common NVIDIA GPUs
            "-gencode=arch=compute_70,code=sm_70",  # V100
            "-gencode=arch=compute_75,code=sm_75",  # RTX 20xx, T4
            "-gencode=arch=compute_80,code=sm_80",  # A100
            "-gencode=arch=compute_86,code=sm_86",  # RTX 30xx
            "-gencode=arch=compute_89,code=sm_89",  # RTX 40xx
        ]
        
        # Add CUDA source file
        sources.append("src/bev_pool_cuda.cu")
        
        print("Building with CUDA support")
    else:
        extension = CppExtension
        print("Building CPU-only version (CUDA not available)")
    
    return extension(
        name="bev_pool_ext",
        sources=sources,
        define_macros=define_macros,
        extra_compile_args=extra_compile_args,
    )


if __name__ == "__main__":
    setup(
        name="bev_pool",
        version="1.0.0",
        description="BEV Pool CUDA extension for efficient camera-to-BEV transformation",
        ext_modules=[make_cuda_ext()],
        cmdclass={"build_ext": BuildExtension},
        python_requires=">=3.8",
        install_requires=[
            "torch>=2.0.0",
        ],
    )


