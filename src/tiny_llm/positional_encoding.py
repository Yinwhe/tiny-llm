import mlx.core as mx


class RoPE:
    def __init__(
        self,
        dims: int,
        seq_len: int,
        base: int = 10000,
        traditional: bool = False,
    ):
        assert dims % 2 == 0
        self.dims = dims
        self.seq_len = seq_len
        self.base = base
        self.half_dims = dims // 2
        self.inv_freqs = mx.power(
            base, -mx.arange(0, self.half_dims, dtype=mx.float32) / self.half_dims
        )
        self.freqs = mx.outer(mx.arange(seq_len), self.inv_freqs)
        self.cos_freqs = mx.cos(self.freqs) # shape: (seq_len, half_dims)
        self.sin_freqs = mx.sin(self.freqs) # shape: (seq_len, half_dims)
        self.traditional = traditional

    def __call__(
        self, x: mx.array, offset: list[slice] | slice | None = None
    ) -> mx.array:
        N, L, H, D = x.shape
        if offset is not None:
            assert isinstance(offset, slice), "Currently list[slice] is not supported"
            assert offset.stop - offset.start == L, f"offset must be of length {L}"
        
        cos_basic = self.cos_freqs[:L, :] if offset is None else self.cos_freqs[offset, :]
        sin_basic = self.sin_freqs[:L, :] if offset is None else self.sin_freqs[offset, :]

        if self.traditional:
            x = x.reshape(N, L, H, self.half_dims, 2)
            x1 = x[..., 0]  # shape: (N, L, H, half_dims)
            x2 = x[..., 1]  # shape: (N, L, H, half_dims)
        else:
            x1 = x[..., :self.half_dims]
            x2 = x[..., self.half_dims:]
        cos_basis = cos_basic.reshape(-1, L, 1, self.half_dims)  # shape: (1, L, 1, half_dims)
        sin_basis = sin_basic.reshape(-1, L, 1, self.half_dims)  # shape: (1, L, 1, half_dims)
        real = mx.multiply(x1, cos_basis) - mx.multiply(x2, sin_basis)
        imag = mx.multiply(x2, cos_basis) + mx.multiply(x1, sin_basis)

        if self.traditional:
            y = mx.stack([real, imag], axis=-1)
            y = y.reshape(N, L, H, D)
        else:
            y = mx.concatenate([real, imag], axis=-1)
        return y.astype(x.dtype)
