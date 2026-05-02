from pathlib import Path
import os
import subprocess
import sys

import torch


if __name__ == "__main__":
    src_dir = Path(__file__).parent.resolve()
    build_root = src_dir / "build"
    build_dir = build_root / "tiny_llm_ext_torch_ref._ext"
    package_dir = src_dir / "tiny_llm_ext_torch_ref"

    build_dir.mkdir(parents=True, exist_ok=True)
    package_dir.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env.setdefault("CMAKE_BUILD_TYPE", "Release")

    # Prefer an explicit architecture list when available. Otherwise compile
    # only for the compute capabilities of the visible CUDA devices.
    explicit_arch_list = env.get("TORCH_CUDA_ARCH_LIST") or env.get("TINY_LLM_CUDA_ARCH_LIST")
    if explicit_arch_list:
        env["TORCH_CUDA_ARCH_LIST"] = explicit_arch_list
    else:
        if torch.version.cuda is None:
            raise RuntimeError(
                "CUDA build requires a CUDA-enabled PyTorch installation, or set "
                "TORCH_CUDA_ARCH_LIST / TINY_LLM_CUDA_ARCH_LIST explicitly."
            )
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA build could not detect a visible GPU. Set "
                "TORCH_CUDA_ARCH_LIST or TINY_LLM_CUDA_ARCH_LIST explicitly."
            )

        detected_arches: list[str] = []
        for device_idx in range(torch.cuda.device_count()):
            major, minor = torch.cuda.get_device_capability(device_idx)
            arch = f"{major}.{minor}"
            if arch not in detected_arches:
                detected_arches.append(arch)
        env["TORCH_CUDA_ARCH_LIST"] = ";".join(detected_arches)

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
