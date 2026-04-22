import shutil

import torch


with torch.no_grad():
    cpu_left = torch.tensor([1, 2, 3], device="cpu")
    cpu_right = torch.tensor([4, 5, 6], device="cpu")
    print(torch.add(cpu_left, cpu_right))

if not torch.cuda.is_available():
    raise RuntimeError("PyTorch cannot find a CUDA GPU")

with torch.no_grad():
    device = torch.device("cuda")
    cuda_left = torch.tensor([1, 2, 3], device=device)
    cuda_right = torch.tensor([4, 5, 6], device=device)
    print(torch.add(cuda_left, cuda_right))

print(f"torch: {torch.__version__}")
print(f"torch cuda: {torch.version.cuda}")
print(f"cuda device: {torch.cuda.get_device_name(0)}")
print(f"nvcc: {shutil.which('nvcc')}")
