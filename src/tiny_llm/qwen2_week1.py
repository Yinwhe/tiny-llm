import mlx.core as mx
from .basics import linear, silu
from .attention import scaled_dot_product_attention_grouped
from .layer_norm import RMSNorm
from .positional_encoding import RoPE
from typing import Any
from .embedding import Embedding
from .quantize import dequantize_linear


class Qwen2MultiHeadAttention:
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        wq: mx.array,
        wk: mx.array,
        wv: mx.array,
        wo: mx.array,
        bq: mx.array,
        bk: mx.array,
        bv: mx.array,
        max_seq_len: int = 32768,
        theta: int = 1000000,
    ):
        assert (
            hidden_size % num_heads == 0
        ), f"hidden_size {hidden_size} must be divisible by num_heads {num_heads}"
        assert (
            num_heads % num_kv_heads == 0
        ), f"num_heads {num_heads} must be divisible by num_kv_heads {num_kv_heads}"
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.num_kv_heads = num_kv_heads
        self.scale = mx.rsqrt(self.head_dim)
        self.wq = wq
        self.wk = wk
        self.wv = wv
        self.wo = wo
        self.bq = bq
        self.bk = bk
        self.bv = bv
        self.rope = RoPE(self.head_dim, max_seq_len, base=theta)

    def __call__(
        self,
        x: mx.array,
        offset: int,
        mask: mx.array | str | None = None,
    ) -> mx.array:
        B, L, E = x.shape
        # Linear -> B, L, Hq, D
        q = linear(x, self.wq, self.bq).reshape(B, L, self.num_heads, self.head_dim)
        # Linear -> B, L, H, D
        k = linear(x, self.wk, self.bk).reshape(B, L, self.num_kv_heads, self.head_dim)
        v = linear(x, self.wv, self.bv).reshape(B, L, self.num_kv_heads, self.head_dim)
        # RoPE
        q = self.rope(q, slice(offset, offset + L)).transpose(0, 2, 1, 3)
        k = self.rope(k, slice(offset, offset + L)).transpose(0, 2, 1, 3)
        v = v.transpose(0, 2, 1, 3)
        # Attention
        x = scaled_dot_product_attention_grouped(q, k, v, self.scale, mask)
        # Linear -> B, L, E
        x = x.transpose(0, 2, 1, 3).reshape(B, L, E)
        return linear(x, self.wo)


class Qwen2MLP:
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        w_gate: mx.array,
        w_up: mx.array,
        w_down: mx.array,
    ):
        self.dim = dim
        self.hidden_dim = hidden_dim
        self.w_gate = w_gate
        self.w_up = w_up
        self.w_down = w_down

    def __call__(self, x: mx.array) -> mx.array:
        return linear(silu(linear(x, self.w_gate)) * linear(x, self.w_up), self.w_down)


class Qwen2TransformerBlock:
    def __init__(
        self,
        num_attention_heads: int,
        num_kv_heads: int,
        hidden_size: int,
        intermediate_size: int,
        rms_norm_eps: float,
        wq: mx.array,
        wk: mx.array,
        wv: mx.array,
        wo: mx.array,
        bq: mx.array,
        bk: mx.array,
        bv: mx.array,
        w_gate: mx.array,
        w_up: mx.array,
        w_down: mx.array,
        w_input_layernorm: mx.array,
        w_post_attention_layernorm: mx.array,
        max_seq_len: int = 32768,
        theta: int = 1000000,
    ):
        assert (
            hidden_size % num_attention_heads == 0
        ), f"hidden_size {hidden_size} must be divisible by num_attention_heads {num_attention_heads}"
        assert (
            num_attention_heads % num_kv_heads == 0
        ), f"num_attention_heads {num_attention_heads} must be divisible by num_kv_heads {num_kv_heads}"

        self.MHA = Qwen2MultiHeadAttention(
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

        self.MLP = Qwen2MLP(
            dim=hidden_size,
            hidden_dim=intermediate_size,
            w_gate=w_gate,
            w_up=w_up,
            w_down=w_down,
        )

        self.InputRMSNorm = RMSNorm(
            dim=hidden_size,
            eps=rms_norm_eps,
            weight=w_input_layernorm,
        )
        self.PostRMSNorm = RMSNorm(
            dim=hidden_size,
            eps=rms_norm_eps,
            weight=w_post_attention_layernorm,
        )

    def __call__(
        self,
        x: mx.array,
        offset: int,
        mask: mx.array | str | None = None,
    ) -> mx.array:
        x += self.MHA(self.InputRMSNorm(x), offset, mask)
        x += self.MLP(self.PostRMSNorm(x))
        return x


class Qwen2ModelWeek1:
    def __init__(self, mlx_model: Any):
        self.num_hidden_layers = mlx_model.args.num_hidden_layers
        self.hidden_size = mlx_model.args.hidden_size
        self.vocab_size = mlx_model.args.vocab_size
        precision = mx.float16
        self.precision = precision

        self.embedding = Embedding(
            vocab_size=self.vocab_size,
            embedding_dim=self.hidden_size,
            weight=dequantize_linear(mlx_model.model.embed_tokens).astype(precision),
        )
        self.layers_inner = []

        for i in range(mlx_model.args.num_hidden_layers):
            wq = dequantize_linear(mlx_model.model.layers[i].self_attn.q_proj)
            wk = dequantize_linear(mlx_model.model.layers[i].self_attn.k_proj)
            wv = dequantize_linear(mlx_model.model.layers[i].self_attn.v_proj)
            wo = dequantize_linear(mlx_model.model.layers[i].self_attn.o_proj)
            w_gate = dequantize_linear(mlx_model.model.layers[i].mlp.gate_proj)
            w_up = dequantize_linear(mlx_model.model.layers[i].mlp.up_proj)
            w_down = dequantize_linear(mlx_model.model.layers[i].mlp.down_proj)

            layer = Qwen2TransformerBlock(
                num_attention_heads=mlx_model.args.num_attention_heads,
                num_kv_heads=mlx_model.args.num_key_value_heads,
                hidden_size=mlx_model.args.hidden_size,
                intermediate_size=mlx_model.args.intermediate_size,
                rms_norm_eps=mlx_model.args.rms_norm_eps,
                wq=wq.astype(precision),
                wk=wk.astype(precision),
                wv=wv.astype(precision),
                wo=wo.astype(precision),
                bq=mlx_model.model.layers[i].self_attn.q_proj.bias.astype(precision),
                bk=mlx_model.model.layers[i].self_attn.k_proj.bias.astype(precision),
                bv=mlx_model.model.layers[i].self_attn.v_proj.bias.astype(precision),
                w_gate=w_gate.astype(precision),
                w_up=w_up.astype(precision),
                w_down=w_down.astype(precision),
                w_input_layernorm=mlx_model.model.layers[
                    i
                ].input_layernorm.weight.astype(precision),
                w_post_attention_layernorm=mlx_model.model.layers[
                    i
                ].post_attention_layernorm.weight.astype(precision),
                max_seq_len=mlx_model.args.max_position_embeddings,
                theta=mlx_model.args.rope_theta,
            )
            self.layers_inner.append(layer)
        self.norm = RMSNorm(
            mlx_model.args.hidden_size,
            weight=mlx_model.model.norm.weight.astype(precision),
            eps=mlx_model.args.rms_norm_eps,
        )
        if not mlx_model.args.tie_word_embeddings:
            self.w_lm_head = dequantize_linear(mlx_model.lm_head)
        else:
            self.w_lm_head = None
        self.mlx_model = mlx_model
        

    def __call__(
        self,
        inputs: mx.array,
        offset: int,
    ) -> mx.array:
        h = self.embedding(inputs)
        for layer in range(self.num_hidden_layers):
            h = self.layers_inner[layer](
                h, offset, mask="causal" if h.shape[1] > 1 else None
            )
        h = self.norm(h)
        if self.w_lm_head is not None:
            return linear(h, self.w_lm_head)
        else:
            return self.embedding.as_linear(h)
