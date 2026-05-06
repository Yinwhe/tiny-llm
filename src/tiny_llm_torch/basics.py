import torch


def softmax(x: torch.Tensor, axis: int) -> torch.Tensor:
    return torch.softmax(x, dim=axis)


def linear(
    x: torch.Tensor,
    w: torch.Tensor,
    bias: torch.Tensor | None = None,
) -> torch.Tensor:
    pass


def silu(x: torch.Tensor) -> torch.Tensor:
    pass
