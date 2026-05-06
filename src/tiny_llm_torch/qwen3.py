from typing import Any

import torch

from .kv_cache import TinyKvCache
from .quantize import QuantizedWeights


class Qwen3MultiHeadAttention:
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        wq: QuantizedWeights,
        wk: QuantizedWeights,
        wv: QuantizedWeights,
        wo: QuantizedWeights,
        q_norm: torch.Tensor,
        k_norm: torch.Tensor,
        max_seq_len: int = 32768,
        theta: int = 1000000,
        rms_norm_eps: float = 1e-5,
        use_flash_attention: bool = False,
    ):
        pass

    def __call__(
        self,
        x: torch.Tensor,
        offsets: list[int],
        cache: TinyKvCache,
        mask: torch.Tensor | str | None = None,
    ) -> torch.Tensor:
        pass


class Qwen3MLP:
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        w_gate: QuantizedWeights,
        w_up: QuantizedWeights,
        w_down: QuantizedWeights,
    ):
        self.dim = dim
        self.hidden_dim = hidden_dim
        self.w_gate = w_gate
        self.w_up = w_up
        self.w_down = w_down

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        pass


class Qwen3TransformerBlock:
    def __init__(
        self,
        num_attention_heads: int,
        num_kv_heads: int,
        hidden_size: int,
        head_dim: int,
        intermediate_size: int,
        rms_norm_eps: float,
        wq: QuantizedWeights,
        wk: QuantizedWeights,
        wv: QuantizedWeights,
        wo: QuantizedWeights,
        q_norm: torch.Tensor,
        k_norm: torch.Tensor,
        w_gate: QuantizedWeights,
        w_up: QuantizedWeights,
        w_down: QuantizedWeights,
        w_input_layernorm: torch.Tensor,
        w_post_attention_layernorm: torch.Tensor,
        max_seq_len: int = 32768,
        theta: int = 1000000,
        use_flash_attention: bool = False,
    ):
        pass

    def __call__(
        self,
        x: torch.Tensor,
        offset: int,
        cache: TinyKvCache,
        mask: torch.Tensor | str | None = None,
    ) -> torch.Tensor:
        pass


def assert_dtype(weights: torch.Tensor, dtype: torch.dtype):
    if weights.dtype != dtype:
        raise ValueError(f"{weights.dtype} != {dtype}")
    else:
        return weights


def assert_quantized_weights_dtype(weights: QuantizedWeights, dtype: torch.dtype):
    if weights.scales.dtype != dtype:
        raise ValueError(f"{weights.scales.dtype} != {dtype}")
    if weights.biases.dtype != dtype:
        raise ValueError(f"{weights.biases.dtype} != {dtype}")
    else:
        return weights


class Qwen3Model:
    def __init__(
        self,
        torch_model: Any,
        enable_flash_attn: bool = False,
    ):
        pass

    def __call__(
        self,
        inputs: torch.Tensor,
        offset: int,
        cache: list[TinyKvCache],
    ) -> torch.Tensor:
        pass
