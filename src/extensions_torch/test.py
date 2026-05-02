from tiny_llm_ext import axpby
import torch

a = torch.ones((3, 4), device="cpu")
b = torch.ones((3, 4), device="cpu")
c = axpby(a, b, 4.0, 2.0)

print(f"c shape: {c.shape}")
print(f"c dtype: {c.dtype}")
print(f"c correct: {torch.all(c == 6.0).item()}")
