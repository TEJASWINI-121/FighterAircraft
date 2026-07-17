import ctypes
import os

libs = [
    'libcudart.so.12',
    'libcublas.so.12',
    'libcublasLt.so.12',
    'libcufft.so.11',
    'libcurand.so.10',
    'libcusolver.so.11',
    'libcusparse.so.12',
    'libcudnn.so.9',
    'libnccl.so.2',
    'libnvJitLink.so.12',
    'libcuda.so.1'
]

for lib in libs:
    try:
        ctypes.cdll.LoadLibrary(lib)
        print(f"SUCCESS: {lib}")
    except OSError as e:
        print(f"FAILED: {lib} - {e}")
