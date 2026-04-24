import torch


class RoPE:
    def __init__(
        self,
        dims: int,
        seq_len: int,
        base: int = 10000,
        traditional: bool = False,
        device: torch.device | str | None = None,
    ):
        pass

    def __call__(
        self, x: torch.Tensor, offset: list[slice] | slice | None = None
    ) -> torch.Tensor:
        pass
