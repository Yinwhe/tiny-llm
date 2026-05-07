import torch


class RMSNorm:
    def __init__(self, dim: int, weight: torch.Tensor, eps: float = 1e-5):
        self.dim = dim
        self.eps = eps
        self.weight = weight.to(dtype=torch.float32)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """
        Shapes:
        - `x`: [B, L, E]
        - returns: [B, L, E]
        """
        original_dtype = x.dtype
        x = x.to(dtype=torch.float32)
        return (
            self.weight
            * x
            * torch.rsqrt(torch.mean(torch.square(x), dim=-1, keepdim=True) + self.eps)
        ).to(dtype=original_dtype)
