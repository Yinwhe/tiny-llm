from pathlib import Path
import os
import subprocess
import sys

import torch


if __name__ == "__main__":
    src_dir = Path(__file__).parent.resolve()
    build_root = src_dir / "build"
    build_dir = build_root / "tiny_llm_ext._ext"
    package_dir = src_dir / "tiny_llm_ext"

    build_dir.mkdir(parents=True, exist_ok=True)
    package_dir.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env.setdefault("CMAKE_BUILD_TYPE", "Release")
    if "TORCH_CUDA_ARCH_LIST" not in env and torch.cuda.is_available():
        major, minor = torch.cuda.get_device_capability(0)
        env["TORCH_CUDA_ARCH_LIST"] = f"{major}.{minor}"

    torch_dir = Path(torch.utils.cmake_prefix_path) / "Torch"

    configure_cmd = [
        "cmake",
        "-S",
        str(src_dir),
        "-B",
        str(build_dir),
        f"-DPython_EXECUTABLE={sys.executable}",
        f"-DTorch_DIR={torch_dir}",
        f"-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={package_dir}",
    ]

    subprocess.run(configure_cmd, check=True, cwd=src_dir, env=env)
    subprocess.run(
        ["cmake", "--build", str(build_dir), "--config", env["CMAKE_BUILD_TYPE"]],
        check=True,
        cwd=src_dir,
        env=env,
    )
