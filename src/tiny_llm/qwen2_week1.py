from typing import Any

import torch


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
        pass

    def __call__(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | str | None = None,
    ) -> torch.Tensor:
        pass


class Qwen2MLP:
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        w_gate: torch.Tensor,
        w_up: torch.Tensor,
        w_down: torch.Tensor,
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
        wq: torch.Tensor,
        wk: torch.Tensor,
        wv: torch.Tensor,
        wo: torch.Tensor,
        bq: torch.Tensor,
        bk: torch.Tensor,
        bv: torch.Tensor,
        w_gate: torch.Tensor,
        w_up: torch.Tensor,
        w_down: torch.Tensor,
        w_input_layernorm: torch.Tensor,
        w_post_attention_layernorm: torch.Tensor,
        max_seq_len: int = 32768,
        theta: int = 1000000,
    ):
        pass

    def __call__(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | str | None = None,
    ) -> torch.Tensor:
        pass


class Qwen2ModelWeek1:
    def __init__(
        self,
        transformers_model: Any,
        precision: torch.dtype = torch.float16,
        device: torch.device | str | None = None,
    ):
        pass

    def __call__(self, inputs: torch.Tensor) -> torch.Tensor:
        pass
