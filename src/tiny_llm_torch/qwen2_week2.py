from typing import Any

import torch

from .kv_cache import TinyKvCache
from .quantize import QuantizedWeights


class Qwen2MultiHeadAttention:
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        wq: QuantizedWeights,
        wk: QuantizedWeights,
        wv: QuantizedWeights,
        wo: QuantizedWeights,
        bq: torch.Tensor,
        bk: torch.Tensor,
        bv: torch.Tensor,
        max_seq_len: int = 32768,
        theta: int = 1000000,
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


class Qwen2MLP:
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        w_gate: QuantizedWeights,
        w_up: QuantizedWeights,
        w_down: QuantizedWeights,
    ):
        pass

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        pass


class Qwen2TransformerBlock:
    def __init__(
        self,
        num_attention_heads: int,
        num_kv_heads: int,
        hidden_size: int,
        intermediate_size: int,
        rms_norm_eps: float,
        wq: QuantizedWeights,
        wk: QuantizedWeights,
        wv: QuantizedWeights,
        wo: QuantizedWeights,
        bq: torch.Tensor,
        bk: torch.Tensor,
        bv: torch.Tensor,
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


class Qwen2ModelWeek2:
    def __init__(
        self,
        torch_model: Any,
        enable_flash_attn: bool = False,
    ):
        self.num_hidden_layers = torch_model.config.num_hidden_layers
        pass

    def __call__(
        self,
        inputs: torch.Tensor,
        offset: int,
        cache: list[TinyKvCache],
    ) -> torch.Tensor:
        pass
