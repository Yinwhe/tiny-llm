from pathlib import Path

import torch

try:
    from ._ext import *

    current_path = Path(__file__).parent
    load_library(
        "cuda" if torch.cuda.is_available() else "cpu",
        str(current_path),
    )
except ImportError:
    print("Failed to load C++/CUDA extension")
