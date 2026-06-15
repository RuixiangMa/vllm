import glob
import importlib.util
import os
import sys

from setuptools import setup
from torch.utils import cpp_extension


def _find_nvcomp(nvcomp_root=None):
    def probe(root):
        inc_dirs = [
            d for d in (
                os.path.join(root, "include"),
                os.path.join(root, "build", "include"),
            ) if os.path.isdir(d)
        ]
        if not any(os.path.exists(os.path.join(d, "nvcomp", "ans.h"))
                   for d in inc_dirs):
            return None
        for sub in ("build/lib", "lib/x86_64-linux-gnu", "lib64", "lib", ""):
            d = os.path.join(root, sub) if sub else root
            if not os.path.isdir(d):
                continue
            if os.path.exists(os.path.join(d, "libnvcomp.so")):
                return inc_dirs, d, "nvcomp"
            versioned = sorted(glob.glob(os.path.join(d, "libnvcomp.so.*")))
            if versioned:
                return inc_dirs, d, ":" + os.path.basename(versioned[-1])
        return None

    if nvcomp_root:
        if not os.path.exists(nvcomp_root):
            raise ValueError(f"NVCOMP_ROOT={nvcomp_root} does not exist")
        result = probe(nvcomp_root)
        if not result:
            raise ValueError(
                f"NVCOMP_ROOT={nvcomp_root} does not contain a usable nvcomp")
        return (*result, f"NVCOMP_ROOT={nvcomp_root}")

    spec = importlib.util.find_spec("nvidia.nvcomp")
    if spec and spec.origin:
        pip_root = os.path.dirname(spec.origin)
        result = probe(pip_root)
        if result:
            return (*result, f"pip nvidia-nvcomp-cu12 ({pip_root})")

    result = probe("/usr")
    if result:
        return (*result, "system (/usr)")

    raise ValueError(
        "nvcomp not found. Install via:\n"
        "  pip install nvidia-nvcomp-cu12\n"
        "Or set NVCOMP_ROOT=/path/to/nvcomp")


def detect_cuda_arch():
    try:
        import torch
        if torch.cuda.is_available():
            archs = set()
            for i in range(torch.cuda.device_count()):
                major, minor = torch.cuda.get_device_capability(i)
                archs.add(f"{major}.{minor}")
            if archs:
                return ";".join(sorted(archs))
    except Exception:
        pass
    return "8.0;8.6;9.0"


if not os.environ.get("TORCH_CUDA_ARCH_LIST"):
    os.environ["TORCH_CUDA_ARCH_LIST"] = detect_cuda_arch()

nvcomp_inc_dirs, nvcomp_lib_dir, nvcomp_link_name, nvcomp_source =     _find_nvcomp(os.environ.get("NVCOMP_ROOT"))

print(f"nvcomp found: source={nvcomp_source}, lib={nvcomp_lib_dir}")

setup(
    name="vllm_ans_transfer",
    ext_modules=[
        cpp_extension.CUDAExtension(
            name="vllm._ans_transfer",
            sources=[
                "csrc/ans_transfer/ans_transfer.cu",
                "csrc/ans_transfer/ans_bindings.cpp",
            ],
            include_dirs=nvcomp_inc_dirs,
            extra_compile_args={
                "nvcc": ["-O3"],
                "cxx": ["-O3", "-std=c++17"],
            },
            extra_link_args=[
                f"-l{nvcomp_link_name}",
                f"-L{nvcomp_lib_dir}",
                f"-Wl,-rpath,{nvcomp_lib_dir}",
            ],
        ),
    ],
    cmdclass={"build_ext": cpp_extension.BuildExtension},
)
