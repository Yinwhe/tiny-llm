import math
from functools import lru_cache

import torch

from extensions_torch_ref import tiny_llm_ext_torch_ref

from .basics import linear, softmax


@lru_cache(maxsize=None)
def _cached_causal_mask(
    L: int,
    S: int,
    device_type: str,
    device_index: int | None,
) -> torch.Tensor:
    device = torch.device(device_type, device_index) if device_index is not None else torch.device(device_type)
    return causal_mask(L, S, torch.float32, device)


def scaled_dot_product_attention_simple(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    scale: float | None = None,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    A simple implementation of scaled dot product attention.

    Shapes:
    - `query`: [B, H, L, D]
    - `key`: [B, H, S, D]
    - `value`: [B, H, S, D]
    - `mask`: broadcastable to [B, H, L, S]
    - returns: [B, H, L, D]
    """
    factor = 1.0 / math.sqrt(query.shape[-1]) if scale is None else scale
    scores = torch.matmul(query, key.swapaxes(-2, -1)) * factor
    if mask is not None:
        scores = scores + mask
    return torch.matmul(softmax(scores, axis=-1), value)


def causal_mask(
    L: int, S: int, dtype: torch.dtype, device: torch.device
) -> torch.Tensor:
    """
    Create a causal mask for attention scores.

    Shapes:
    - returns: [L, S]
    """
    mask = torch.tril(
        torch.ones((L, S), device=device, dtype=torch.bool),
        diagonal=S - L,
    )
    return torch.where(mask, 0.0, -torch.inf).to(dtype=dtype)


def scaled_dot_product_attention_grouped(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    scale: float | None = None,
    mask: torch.Tensor | str | None = None,
) -> torch.Tensor:
    """
    Grouped-query attention.

    Shapes:
    - `query`: [B, H_q, L, D]
    - `key`: [B, H_kv, S, D]
    - `value`: [B, H_kv, S, D]
    - `mask`: `None`, `"causal"`, or tensor broadcastable to [B, H_q, L, S]
    - returns: [B, H_q, L, D]
    """
    expected_shape = query.shape
    *batch_shape, H_q, L, D = query.shape
    H, S, _ = key.shape[-3:]

    assert H_q % H == 0, "H_q must be divisible by H"
    repeats = H_q // H

    factor = 1.0 / math.sqrt(D) if scale is None else scale
    factor = query.new_tensor(factor)

    query = query.reshape(*batch_shape, H, repeats, L, D)
    key = key.reshape(*batch_shape, H, 1, S, D)
    value = value.reshape(*batch_shape, H, 1, S, D)

    scores = (
        torch.matmul(
            query,
            key.transpose(-2, -1),
        )
        * factor
    )
    if mask == "causal":
        mask = causal_mask(L, S, scores.dtype, scores.device)
        scores = scores + mask
    elif isinstance(mask, torch.Tensor):
        scores = scores + mask.reshape(*batch_shape, H, repeats, L, S)
    elif mask is not None:
        raise NotImplementedError

    return torch.matmul(softmax(scores, axis=-1), value).reshape(expected_shape)


def flash_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    scale: float | None = None,
    mask: torch.Tensor | str | None = None,
) -> torch.Tensor:
    """
    Flash attention wrapper around the C++/CUDA extension.

    Shapes:
    - `query`: [B..., H_q, L, E]
    - `key`: [B..., H_kv, S, E]
    - `value`: [B..., H_kv, S, E]
    - `mask`: `None`, `"causal"`, or tensor broadcastable to [B..., H_q, L, S]
    - returns: [B..., H_q, L, E]
    """
    expected_shape = query.shape
    *batch_shape, H_q, L, E = query.shape
    _, H, S, _ = key.shape

    factor = 1.0 / math.sqrt(E) if scale is None else float(scale)

    query = query.reshape(-1, L, E).contiguous()
    key = key.reshape(-1, S, E).contiguous()
    value = value.reshape(-1, S, E).contiguous()

    is_causal = isinstance(mask, str) and mask == "causal"
    N = query.shape[0]

    if is_causal:
        mask = torch.broadcast_to(
            _cached_causal_mask(L, S, query.device.type, query.device.index),
            (*batch_shape, H_q, L, S),
        )
    elif mask is None:
        mask = torch.broadcast_to(
            torch.zeros((L, S), dtype=torch.float32, device=query.device),
            (*batch_shape, H_q, L, S),
        )
    else:
        mask = torch.broadcast_to(mask, (*batch_shape, H_q, L, S))

    mask = mask.reshape(N, L, S).contiguous().to(dtype=torch.float32)

    result = tiny_llm_ext_torch_ref.flash_attention(
        query,
        key,
        value,
        mask,
        factor,
        is_causal=is_causal,
        num_heads=H_q,
        num_kv_heads=H,
    )
    return result.reshape(expected_shape).contiguous()


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
        """
        Shapes:
        - `query`: [B, L, E]
        - `key`: [B, S, E]
        - `value`: [B, S, E]
        - `mask`: broadcastable to [B, H, L, S]
        - returns: [B, L, E]
        """
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
