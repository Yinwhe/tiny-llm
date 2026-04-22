import math

import torch

from .basics import linear, softmax


def scaled_dot_product_attention_simple(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    scale: float | None = None,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    A simple implementation of scaled dot product attention. Assuming Q, K, V are of the same shape.
    Assuming mask is always a float array that you can add to the scores.
    """
    factor = 1.0 / math.sqrt(query.shape[-1]) if scale is None else scale
    scores = torch.matmul(query, key.swapaxes(-2, -1)) * factor
    if mask is not None:
        scores = scores + mask
    return torch.matmul(softmax(scores, axis=-1), value)


class SimpleMultiHeadAttention:
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        wq: torch.Tensor,
        wk: torch.Tensor,
        wv: torch.Tensor,
        wo: torch.Tensor,
    ):
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        assert hidden_size % num_heads == 0
        self.head_dim = hidden_size // num_heads
        self.scale = 1.0 / math.sqrt(self.head_dim)
        assert wq.shape == (num_heads * self.head_dim, hidden_size)
        assert wk.shape == (num_heads * self.head_dim, hidden_size)
        assert wv.shape == (num_heads * self.head_dim, hidden_size)
        assert wo.shape == (hidden_size, num_heads * self.head_dim)
        self.wq = wq
        self.wk = wk
        self.wv = wv
        self.wo = wo

    def __call__(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        N, L, _ = query.shape
        assert query.shape == key.shape == value.shape

        projection_q = (
            linear(query, self.wq)
            .reshape(N, L, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )
        projection_k = (
            linear(key, self.wk)
            .reshape(N, L, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )
        projection_v = (
            linear(value, self.wv)
            .reshape(N, L, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )
        x = scaled_dot_product_attention_simple(
            projection_q,
            projection_k,
            projection_v,
            scale=self.scale,
            mask=mask,
        )
        x = x.transpose(1, 2).reshape(N, L, self.hidden_size)
        return linear(x, self.wo)
