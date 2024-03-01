import glob
import os.path as osp

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

ROOT_DIR = osp.dirname(osp.abspath(__file__))
include_dirs = [osp.join(ROOT_DIR, "include")]

sources = glob.glob("*.cpp") + glob.glob("*.cu")
for dir in ["cuda", "cpu"]:
    sources += glob.glob(f"{dir}/*.cpp")
    sources += glob.glob(f"{dir}/*.cu")

print(sources)
setup(
    name="sparse_gs",
    version="1.0",
    author="Loick Chambon",
    author_email="loick.chambon@valeo.com",
    description="Sparse grid sampling.",
    ext_modules=[
        CUDAExtension(
            name="sparse_gs",
            sources=sources,
            include_dirs=include_dirs,
            extra_compile_args={"cxx": ["-O2"], "nvcc": ["-O2"]},
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
