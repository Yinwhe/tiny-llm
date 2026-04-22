import torch
import torch.nn.functional as F


def softmax(x: torch.Tensor, axis: int) -> torch.Tensor:
    return F.softmax(x, dim=axis)


def linear(
    x: torch.Tensor,
    w: torch.Tensor,
    bias: torch.Tensor | None = None,
) -> torch.Tensor:
    if bias is None:
        return torch.matmul(x, w.T)
    else:
        return torch.matmul(x, w.T) + bias


def silu(x: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.silu(x)
