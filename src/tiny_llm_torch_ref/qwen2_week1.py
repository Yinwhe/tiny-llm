import math

import torch

from .attention import scaled_dot_product_attention_grouped
from .basics import linear
from .positional_encoding import RoPE


class Qwen2MultiHeadAttention:
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        wq: torch.Tensor,
        wk: torch.Tensor,
        wv: torch.Tensor,
        wo: torch.Tensor,
        bq: torch.Tensor,
        bk: torch.Tensor,
        bv: torch.Tensor,
        max_seq_len: int = 32768,
        theta: int = 1000000,
    ):
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        assert hidden_size % num_heads == 0, (
            f"hidden_size {hidden_size} must be divisible by num_heads {num_heads}"
        )
        assert num_heads % num_kv_heads == 0, (
            f"num_heads {num_heads} must be divisible by num_kv_heads {num_kv_heads}"
        )

        self.head_dim = hidden_size // num_heads
        self.scale = 1.0 / math.sqrt(self.head_dim)
        self.wq = wq
        self.wk = wk
        self.wv = wv
        self.wo = wo
        self.bq = bq
        self.bk = bk
        self.bv = bv
        self.rope = RoPE(
            self.head_dim,
            max_seq_len,
            theta,
            traditional=False,
            device=wq.device,
        )

    def __call__(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | str | None = None,
    ) -> torch.Tensor:
        batch_size, seq_len, hidden_size = x.shape

        projection_q = linear(x, self.wq, self.bq).reshape(
            batch_size, seq_len, self.num_heads, self.head_dim
        )
        projection_k = linear(x, self.wk, self.bk).reshape(
            batch_size, seq_len, self.num_kv_heads, self.head_dim
        )
        projection_v = linear(x, self.wv, self.bv).reshape(
            batch_size, seq_len, self.num_kv_heads, self.head_dim
        )

        projection_q = self.rope(projection_q, offset=slice(0, seq_len))
        projection_k = self.rope(projection_k, offset=slice(0, seq_len))

        output = scaled_dot_product_attention_grouped(
            projection_q.transpose(1, 2).to(dtype=torch.float32),
            projection_k.transpose(1, 2).to(dtype=torch.float32),
            projection_v.transpose(1, 2).to(dtype=torch.float32),
            scale=self.scale,
            mask=mask,
        ).to(dtype=x.dtype)
        output = output.transpose(1, 2).reshape(batch_size, seq_len, hidden_size)

        return linear(output, self.wo)
