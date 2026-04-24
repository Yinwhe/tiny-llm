import torch
import torch.nn.functional as F


def softmax(x: torch.Tensor, axis: int) -> torch.Tensor:
    """
    Shapes:
    - `x`: arbitrary tensor
    - returns: same shape as `x`
    """
    return F.softmax(x, dim=axis)


def linear(
    x: torch.Tensor,
    w: torch.Tensor,
    bias: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Shapes:
    - `x`: [..., in_features]
    - `w`: [out_features, in_features]
    - `bias`: [out_features]
    - returns: [..., out_features]
    """
    if bias is None:
        return torch.matmul(x, w.T)
    else:
        return torch.matmul(x, w.T) + bias


def silu(x: torch.Tensor) -> torch.Tensor:
    """
    Shapes:
    - `x`: arbitrary tensor
    - returns: same shape as `x`
    """
    return x * torch.sigmoid(x)
