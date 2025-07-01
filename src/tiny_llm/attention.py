import mlx.core as mx
from .basics import softmax, linear


def scaled_dot_product_attention_simple(
    query: mx.array,
    key: mx.array,
    value: mx.array,
    scale: float | None = None,
    mask: mx.array | None = None,
) -> mx.array:
    factor = mx.rsqrt(query.shape[-1]) if scale is None else scale
    qk_mat = mx.matmul(query, key.swapaxes(-2, -1)) * factor
    if mask is not None:
        qk_mat += mask
    return mx.matmul(softmax(qk_mat, -1), value)


class SimpleMultiHeadAttention:
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        wq: mx.array,
        wk: mx.array,
        wv: mx.array,
        wo: mx.array,
    ):
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        assert self.hidden_size % self.num_heads == 0
        self.head_dim = hidden_size // num_heads
        assert wq.shape == (hidden_size, num_heads * self.head_dim)
        assert wk.shape == (hidden_size, num_heads * self.head_dim)
        assert wv.shape == (hidden_size, num_heads * self.head_dim)
        assert wo.shape == (num_heads * self.head_dim, hidden_size)
        self.wq = wq
        self.wk = wk
        self.wv = wv
        self.wo = wo
        self.scale = mx.rsqrt(mx.array(self.head_dim, dtype=mx.float32))

    def __call__(
        self,
        query: mx.array,
        key: mx.array,
        value: mx.array,
        mask: mx.array | None = None,
    ) -> mx.array:
        N, L, _ = query.shape
        # Reshape query, key, value to N * L * H * D and swap axes to N * H * L * D
        q_proj = (
            linear(query, self.wq)
            .reshape(N, L, self.num_heads, self.head_dim)
            .swapaxes(1, 2)
        )
        k_proj = (
            linear(key, self.wk)
            .reshape(N, L, self.num_heads, self.head_dim)
            .swapaxes(1, 2)
        )
        v_proj = (
            linear(value, self.wv)
            .reshape(N, L, self.num_heads, self.head_dim)
            .swapaxes(1, 2)
        )
        # attn is N * H * L * D and reshape/swapaxes to N * L * H * D
        attn = (
            scaled_dot_product_attention_simple(
                q_proj, k_proj, v_proj, float(self.scale), mask
            )
            .swapaxes(1, 2)
            .reshape(N, L, self.num_heads * self.head_dim)
        )

        return linear(attn, self.wo)


def causal_mask(L: int, S: int, dtype: mx.Dtype) -> mx.array:
    mask = mx.tril(mx.ones((L, S)), k=S - L)
    return mx.where(mask, 0.0, -mx.inf).astype(dtype)


def scaled_dot_product_attention_grouped(
    query: mx.array,
    key: mx.array,
    value: mx.array,
    scale: float | None = None,
    mask: mx.array | str | None = None,
) -> mx.array:
    factor = mx.rsqrt(query.shape[-1]) if scale is None else scale
    expected_shape = query.shape

    H_q, L, D = query.shape[-3:]
    H, S, _ = key.shape[-3:]
    assert H_q % H == 0

    # Reshape to group heads
    query = query.reshape(-1, H, H_q // H, L, D)
    key = key.reshape(-1, H, 1, S, D)
    value = value.reshape(-1, H, 1, S, D)

    # Apply attention
    qk_mat = mx.matmul(query, key.swapaxes(-2, -1)) * factor
    if isinstance(mask, str):
        if mask == "causal":
            qk_mat += causal_mask(L, S, query.dtype)
        else:
            raise ValueError(f"Invalid mask: {mask}")
    elif mask is not None:
        qk_mat += mask.reshape(-1, H, H_q // H, L, S)
    return mx.matmul(softmax(qk_mat, -1), value).reshape(expected_shape)


def flash_attention(
    query: mx.array,
    key: mx.array,
    value: mx.array,
    scale: float | None = None,
) -> mx.array:
    pass
