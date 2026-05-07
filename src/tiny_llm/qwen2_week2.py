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
        wq: torch.Tensor | QuantizedWeights,
        wk: torch.Tensor | QuantizedWeights,
        wv: torch.Tensor | QuantizedWeights,
        wo: torch.Tensor | QuantizedWeights,
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
        offsets: int | list[int] | torch.Tensor,
        cache: TinyKvCache,
        mask: torch.Tensor | str | None = None,
    ) -> torch.Tensor:
        """
        Shapes:
        - `x`: [B, L_new, E]
        - `offsets`: one start position per batch item, or a scalar shared by all
        - `mask`: `None`, `"causal"`, or tensor broadcastable to [B, H_q, L_new, L_total]
        - returns: [B, L_new, E]
        """
        pass


class Qwen2MLP:
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        w_gate: torch.Tensor | QuantizedWeights,
        w_up: torch.Tensor | QuantizedWeights,
        w_down: torch.Tensor | QuantizedWeights,
    ):
        pass

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """
        Shapes:
        - `x`: [B, L, E]
        - returns: [B, L, E]
        """
        pass


class Qwen2TransformerBlock:
    def __init__(
        self,
        num_attention_heads: int,
        num_kv_heads: int,
        hidden_size: int,
        intermediate_size: int,
        rms_norm_eps: float,
        wq: torch.Tensor | QuantizedWeights,
        wk: torch.Tensor | QuantizedWeights,
        wv: torch.Tensor | QuantizedWeights,
        wo: torch.Tensor | QuantizedWeights,
        bq: torch.Tensor,
        bk: torch.Tensor,
        bv: torch.Tensor,
        w_gate: torch.Tensor | QuantizedWeights,
        w_up: torch.Tensor | QuantizedWeights,
        w_down: torch.Tensor | QuantizedWeights,
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
        offset: int | list[int] | torch.Tensor,
        cache: TinyKvCache,
        mask: torch.Tensor | str | None = None,
    ) -> torch.Tensor:
        """
        Shapes:
        - `x`: [B, L_new, E]
        - `offset`: one start position per batch item, or a scalar shared by all
        - returns: [B, L_new, E]
        """
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
        offset: int | list[int] | torch.Tensor,
        cache: list[TinyKvCache],
    ) -> torch.Tensor:
        """
        Shapes:
        - `inputs`: [B, L_new]
        - returns logits: [B, L_new, V]
        """
        pass
