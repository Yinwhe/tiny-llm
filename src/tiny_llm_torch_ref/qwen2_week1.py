import math
from typing import Any

import torch

from .attention import scaled_dot_product_attention_grouped
from .basics import linear, silu
from .embedding import Embedding
from .layer_norm import RMSNorm
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


class Qwen2MLP:
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        w_gate: torch.Tensor,
        w_up: torch.Tensor,
        w_down: torch.Tensor,
    ):
        self.dim = dim
        self.hidden_dim = hidden_dim
        self.w_gate = w_gate
        self.w_up = w_up
        self.w_down = w_down

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return linear(silu(linear(x, self.w_gate)) * linear(x, self.w_up), self.w_down)


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
        self.num_attention_heads = num_attention_heads
        self.hidden_size = hidden_size
        self.mlp = Qwen2MLP(hidden_size, intermediate_size, w_gate, w_up, w_down)
        self.input_layernorm = RMSNorm(hidden_size, w_input_layernorm, eps=rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(
            hidden_size,
            w_post_attention_layernorm,
            eps=rms_norm_eps,
        )
        self.self_attn = Qwen2MultiHeadAttention(
            hidden_size=hidden_size,
            num_heads=num_attention_heads,
            num_kv_heads=num_kv_heads,
            wq=wq,
            wk=wk,
            wv=wv,
            wo=wo,
            bq=bq,
            bk=bk,
            bv=bv,
            max_seq_len=max_seq_len,
            theta=theta,
        )

    def __call__(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | str | None = None,
    ) -> torch.Tensor:
        residual = self.self_attn(self.input_layernorm(x), mask)
        hidden = x + residual
        residual = self.mlp(self.post_attention_layernorm(hidden))
        return hidden + residual


class Qwen2ModelWeek1:
    def __init__(
        self,
        transformers_model: Any,
        precision: torch.dtype = torch.float16,
        device: torch.device | str | None = None,
    ):
        config = transformers_model.config
        model = transformers_model.model

        if device is None:
            device = model.embed_tokens.weight.device
        device = torch.device(device)

        def cast(weight: torch.Tensor) -> torch.Tensor:
            return weight.detach().to(device=device, dtype=precision).contiguous()

        def cast_bias(
            bias: torch.Tensor | None,
            out_features: int,
        ) -> torch.Tensor:
            if bias is None:
                return torch.zeros(out_features, device=device, dtype=precision)
            return cast(bias)

        self.num_hidden_layers = config.num_hidden_layers
        self.hidden_size = config.hidden_size
        self.vocab_size = config.vocab_size
        self.precision = precision

        self.embedding = Embedding(
            vocab_size=self.vocab_size,
            embedding_dim=self.hidden_size,
            weight=cast(model.embed_tokens.weight),
        )
        self.layers_inner = []

        for index in range(self.num_hidden_layers):
            transformer_layer = model.layers[index]
            attention = transformer_layer.self_attn
            mlp = transformer_layer.mlp

            layer = Qwen2TransformerBlock(
                num_attention_heads=config.num_attention_heads,
                num_kv_heads=config.num_key_value_heads,
                hidden_size=config.hidden_size,
                intermediate_size=config.intermediate_size,
                rms_norm_eps=config.rms_norm_eps,
                wq=cast(attention.q_proj.weight),
                wk=cast(attention.k_proj.weight),
                wv=cast(attention.v_proj.weight),
                wo=cast(attention.o_proj.weight),
                bq=cast_bias(attention.q_proj.bias, attention.q_proj.out_features),
                bk=cast_bias(attention.k_proj.bias, attention.k_proj.out_features),
                bv=cast_bias(attention.v_proj.bias, attention.v_proj.out_features),
                w_gate=cast(mlp.gate_proj.weight),
                w_up=cast(mlp.up_proj.weight),
                w_down=cast(mlp.down_proj.weight),
                w_input_layernorm=cast(transformer_layer.input_layernorm.weight),
                w_post_attention_layernorm=cast(
                    transformer_layer.post_attention_layernorm.weight
                ),
                max_seq_len=config.max_position_embeddings,
                theta=config.rope_theta,
            )
            self.layers_inner.append(layer)

        self.norm = RMSNorm(
            self.hidden_size,
            weight=cast(model.norm.weight),
            eps=config.rms_norm_eps,
        )
        if config.tie_word_embeddings:
            self.w_lm_head = None
        else:
            self.w_lm_head = cast(transformers_model.lm_head.weight)

    def __call__(self, inputs: torch.Tensor) -> torch.Tensor:
        hidden = self.embedding(inputs)
        for index in range(self.num_hidden_layers):
            hidden = self.layers_inner[index](hidden, mask="causal")
        hidden = self.norm(hidden)
        if self.w_lm_head is not None:
            return linear(hidden, self.w_lm_head)
        return self.embedding.as_linear(hidden)
