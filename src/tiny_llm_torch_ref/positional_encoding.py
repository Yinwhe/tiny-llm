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
        assert dims % 2 == 0, "dims must be even"

        self.dims = dims
        self.seq_len = seq_len
        self.base = base
        self.traditional = traditional
        self.half_dims = dims // 2
        self.device = (
            torch.device(device)
            if device is not None
            else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )
        if self.device.type == "cuda" and self.device.index is None:
            self.device = torch.device(f"cuda:{torch.cuda.current_device()}")

        idx = torch.arange(0, dims, 2, device=self.device, dtype=torch.float32)
        pos = torch.arange(seq_len, device=self.device, dtype=torch.float32)

        freq = 1.0 / (base ** (idx / dims))
        angles = pos[:, None] * freq[None, :]

        self.cos = torch.cos(angles)[None, :, None, :, None]
        self.sin = torch.sin(angles)[None, :, None, :, None]

    def __call__(
        self, x: torch.Tensor, offset: list[slice] | slice | None = None
    ) -> torch.Tensor:
        """
        Shapes:
        - `x`: [B, L, H, D]
        - returns: [B, L, H, D]
        """
        N, L, H, D = x.shape
        assert D == self.dims, f"input last dim must be {self.dims}, got {D}"
        assert x.device == self.device, f"input must be on {self.device}, got {x.device}"

        if offset is None:
            cos = self.cos[:, :L]
            sin = self.sin[:, :L]
        elif isinstance(offset, slice):
            assert offset.start is not None
            assert offset.stop is not None
            assert offset.stop - offset.start == L, f"offset must be of length {L}"
            cos = self.cos[:, offset]
            sin = self.sin[:, offset]
        else:
            raise NotImplementedError

        if self.traditional:
            x = x.reshape(N, L, H, self.half_dims, 2)
            x1 = x[..., 0]
            x2 = x[..., 1]
        else:
            x1 = x[..., : self.half_dims]
            x2 = x[..., self.half_dims :]

        y1 = x1 * cos[..., 0] - x2 * sin[..., 0]
        y2 = x1 * sin[..., 0] + x2 * cos[..., 0]

        if self.traditional:
            y = torch.stack([y1, y2], dim=-1)
        else:
            y = torch.cat([y1, y2], dim=-1)

        y = y.reshape(N, L, H, D)
        return y.to(dtype=x.dtype)
