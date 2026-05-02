import math
from typing import Any

import torch

from .attention import scaled_dot_product_attention_grouped
from .basics import linear, silu
from .embedding import Embedding
from .kv_cache import TinyKvCache
from .layer_norm import RMSNorm
from .positional_encoding import RoPE
from .quantize import QuantizedWeights, quantized_linear


def apply_linear(
    x: torch.Tensor,
    weight: torch.Tensor | QuantizedWeights,
    bias: torch.Tensor | None = None,
) -> torch.Tensor:
    if isinstance(weight, QuantizedWeights):
        return quantized_linear(x, weight, bias)
    return linear(x, weight, bias)


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
        self.use_flash_attention = use_flash_attention
        self.rope = RoPE(
            self.head_dim,
            max_seq_len,
            theta,
            traditional=False,
            device=wq.scales.device if isinstance(wq, QuantizedWeights) else wq.device,
        )

    def __call__(
        self,
        x: torch.Tensor,
        offset: int,
        cache: TinyKvCache,
        mask: torch.Tensor | str | None = None,
    ) -> torch.Tensor:
        """
        Shapes:
        - `x`: [B, L_new, E]
        - `offset`: scalar start position of this chunk
        - `mask`: `None`, `"causal"`, or tensor broadcastable to [B, H_q, L_new, L_total]
        - returns: [B, L_new, E]
        """
        batch_size, seq_len, _ = x.shape
        if hasattr(cache, "offset"):
            assert cache.offset == offset, (
                f"cache offset {cache.offset} must match input offset {offset}"
            )

        projection_q = apply_linear(x, self.wq, self.bq).reshape(
            batch_size, seq_len, self.num_heads, self.head_dim
        )
        projection_k = apply_linear(x, self.wk, self.bk).reshape(
            batch_size, seq_len, self.num_kv_heads, self.head_dim
        )
        projection_v = apply_linear(x, self.wv, self.bv).reshape(
            batch_size, seq_len, self.num_kv_heads, self.head_dim
        )

        projection_q = self.rope(projection_q, offset=slice(offset, offset + seq_len))
        projection_k = self.rope(projection_k, offset=slice(offset, offset + seq_len))
        projection_q = projection_q.transpose(1, 2)
        projection_k = projection_k.transpose(1, 2)
        projection_v = projection_v.transpose(1, 2)

        projection_k, projection_v, _, mask = cache.update_and_fetch(
            projection_k,
            projection_v,
            mask_length=seq_len,
            mask=mask,
        )

        output = scaled_dot_product_attention_grouped(
            projection_q.to(dtype=torch.float32),
            projection_k.to(dtype=torch.float32),
            projection_v.to(dtype=torch.float32),
            scale=self.scale,
            mask=mask,
        ).to(dtype=x.dtype)
        output = output.transpose(1, 2).reshape(batch_size, seq_len, self.hidden_size)

        return apply_linear(output, self.wo)


class Qwen2MLP:
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        w_gate: torch.Tensor | QuantizedWeights,
        w_up: torch.Tensor | QuantizedWeights,
        w_down: torch.Tensor | QuantizedWeights,
    ):
        self.dim = dim
        self.hidden_dim = hidden_dim
        self.w_gate = w_gate
        self.w_up = w_up
        self.w_down = w_down

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """
        Shapes:
        - `x`: [B, L, E]
        - returns: [B, L, E]
        """
        return apply_linear(
            silu(apply_linear(x, self.w_gate)) * apply_linear(x, self.w_up),
            self.w_down,
        )


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
        self.num_attention_heads = num_attention_heads
        self.hidden_size = hidden_size
        self.use_flash_attention = use_flash_attention
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
            use_flash_attention=use_flash_attention,
        )

    def __call__(
        self,
        x: torch.Tensor,
        offset: int,
        cache: TinyKvCache,
        mask: torch.Tensor | str | None = None,
    ) -> torch.Tensor:
        """
        Shapes:
        - `x`: [B, L_new, E]
        - `offset`: scalar start position of this chunk
        - returns: [B, L_new, E]
        """
        residual = self.self_attn(self.input_layernorm(x), offset, cache, mask)
        hidden = x + residual
        residual = self.mlp(self.post_attention_layernorm(hidden))
        return hidden + residual


class Qwen2ModelWeek2:
    def __init__(
        self,
        transformers_model: Any,
        precision: torch.dtype = torch.float16,
        device: torch.device | str | None = None,
        enable_flash_attn: bool = False,
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

        def cast_weight(torch_layer: Any) -> torch.Tensor | QuantizedWeights:
            required_attrs = ("qweight", "qzeros", "scales", "group_size", "w_bit")
            if all(hasattr(torch_layer, attr) for attr in required_attrs):
                return QuantizedWeights(
                    scales=torch_layer.scales.detach().to(
                        device=device, dtype=precision
                    ).contiguous(),
                    zeros=torch_layer.qzeros.detach().to(
                        device=device, dtype=torch.int32
                    ).contiguous(),
                    group_size=int(torch_layer.group_size),
                    bits=int(torch_layer.w_bit),
                    weight=torch_layer.qweight.detach().to(
                        device=device, dtype=torch.int32
                    ).contiguous(),
                )
            if hasattr(torch_layer, "weight"):
                return cast(torch_layer.weight)
            raise RuntimeError("unsupported linear layer type for Qwen2ModelWeek2")

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
                wq=cast_weight(attention.q_proj),
                wk=cast_weight(attention.k_proj),
                wv=cast_weight(attention.v_proj),
                wo=cast_weight(attention.o_proj),
                bq=cast_bias(attention.q_proj.bias, attention.q_proj.out_features),
                bk=cast_bias(attention.k_proj.bias, attention.k_proj.out_features),
                bv=cast_bias(attention.v_proj.bias, attention.v_proj.out_features),
                w_gate=cast_weight(mlp.gate_proj),
                w_up=cast_weight(mlp.up_proj),
                w_down=cast_weight(mlp.down_proj),
                w_input_layernorm=cast(transformer_layer.input_layernorm.weight),
                w_post_attention_layernorm=cast(
                    transformer_layer.post_attention_layernorm.weight
                ),
                max_seq_len=config.max_position_embeddings,
                theta=config.rope_theta,
                use_flash_attention=enable_flash_attn,
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
            self.w_lm_head = cast_weight(transformers_model.lm_head)

    def __call__(
        self,
        inputs: torch.Tensor,
        offset: int,
        cache: list[TinyKvCache],
    ) -> torch.Tensor:
        """
        Shapes:
        - `inputs`: [B, L_new]
        - returns logits: [B, L_new, V]
        """
        hidden = self.embedding(inputs)
        for index in range(self.num_hidden_layers):
            hidden = self.layers_inner[index](
                hidden, offset, cache[index], mask="causal"
            )
        hidden = self.norm(hidden)
        if self.w_lm_head is not None:
            return apply_linear(hidden, self.w_lm_head)
        return self.embedding.as_linear(hidden)
