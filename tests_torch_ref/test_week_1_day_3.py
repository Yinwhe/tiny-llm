import math

import pytest
import torch

from .tiny_llm_base import (
    Qwen2MultiHeadAttention,
    RoPE,
    causal_mask,
    linear,
    scaled_dot_product_attention_grouped,
)
from .utils import (
    AVAILABLE_DEVICES,
    AVAILABLE_DEVICES_IDS,
    PRECISION_IDS,
    PRECISIONS,
    assert_allclose,
    make_device,
    rand_uniform,
)


def reference_causal_mask(
    L: int,
    S: int,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    mask = torch.tril(torch.ones((L, S), device=device, dtype=torch.bool), diagonal=S - L)
    return torch.where(mask, torch.tensor(0.0, device=device), -torch.inf).to(dtype=dtype)


def reference_grouped_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    scale: float | None = None,
    mask: torch.Tensor | str | None = None,
) -> torch.Tensor:
    expected_shape = query.shape
    *batch_shape, num_query_heads, query_len, head_dim = query.shape
    num_kv_heads, key_len, _ = key.shape[-3:]
    assert num_query_heads % num_kv_heads == 0

    repeats = num_query_heads // num_kv_heads
    factor = 1.0 / math.sqrt(head_dim) if scale is None else scale

    query = query.reshape(*batch_shape, 1, num_kv_heads, repeats, query_len, head_dim)
    key = key.reshape(*batch_shape, 1, num_kv_heads, 1, key_len, head_dim)
    value = value.reshape(*batch_shape, 1, num_kv_heads, 1, key_len, head_dim)

    scores = torch.matmul(query, key.swapaxes(-2, -1)) * factor
    if mask is not None:
        if mask == "causal":
            scores = scores + reference_causal_mask(
                query_len,
                key_len,
                scores.dtype,
                scores.device,
            )
        else:
            mask = torch.broadcast_to(mask, (*batch_shape, num_query_heads, query_len, key_len))
            mask = mask.reshape(*batch_shape, 1, num_kv_heads, repeats, query_len, key_len)
            scores = scores + mask

    result = torch.matmul(torch.softmax(scores, dim=-1), value)
    return result.reshape(expected_shape).to(dtype=query.dtype)


def grouped_attention_helper(
    device_type: str,
    precision: torch.dtype,
    batch_dimension: int,
    scale: float | None,
    is_causal_mask: bool,
):
    device = make_device(device_type)
    num_query_heads = 18
    num_kv_heads = 6
    query_len = 3
    head_dim = 5
    key_len = 7
    batch = 10
    batch_2 = 2

    if batch_dimension == 0:
        q_shape = (num_query_heads, query_len, head_dim)
        kv_shape = (num_kv_heads, key_len, head_dim)
        mask_shape = (num_query_heads, query_len, key_len)
    elif batch_dimension == 1:
        q_shape = (batch, num_query_heads, query_len, head_dim)
        kv_shape = (batch, num_kv_heads, key_len, head_dim)
        mask_shape = (batch, num_query_heads, query_len, key_len)
    elif batch_dimension == 2:
        q_shape = (batch_2, batch, num_query_heads, query_len, head_dim)
        kv_shape = (batch_2, batch, num_kv_heads, key_len, head_dim)
        mask_shape = (batch_2, batch, num_query_heads, query_len, key_len)
    else:
        raise ValueError(f"Unsupported batch dimension: {batch_dimension}")

    for _ in range(100):
        query = rand_uniform(q_shape, device, precision)
        key = rand_uniform(kv_shape, device, precision)
        value = rand_uniform(kv_shape, device, precision)
        mask = rand_uniform(mask_shape, device, precision)
        mask_arg = "causal" if is_causal_mask else mask

        reference_output = reference_grouped_attention(
            query,
            key,
            value,
            scale=scale,
            mask=mask_arg,
        )
        user_output = scaled_dot_product_attention_grouped(
            query,
            key,
            value,
            scale=scale,
            mask=mask_arg,
        )

        assert_allclose(user_output, reference_output, precision=precision)


@pytest.mark.parametrize("device_type", AVAILABLE_DEVICES, ids=AVAILABLE_DEVICES_IDS)
@pytest.mark.parametrize("precision", PRECISIONS, ids=PRECISION_IDS)
@pytest.mark.parametrize(
    "batch_dimension", [0, 1, 2], ids=["batch_0", "batch_1", "batch_2"]
)
@pytest.mark.parametrize("scale", [None, 0.8])
def test_task_1_grouped_attention(
    device_type: str,
    precision: torch.dtype,
    batch_dimension: int,
    scale: float | None,
):
    grouped_attention_helper(device_type, precision, batch_dimension, scale, False)


@pytest.mark.parametrize("device_type", AVAILABLE_DEVICES, ids=AVAILABLE_DEVICES_IDS)
def test_task_2_mask_only_same_dim(device_type: str):
    device = make_device(device_type)
    user_output = causal_mask(3, 3, torch.float32, device)
    assert_allclose(
        user_output,
        torch.tensor(
            [
                [0, -torch.inf, -torch.inf],
                [0, 0, -torch.inf],
                [0, 0, 0],
            ],
            device=device,
        ),
        precision=torch.float32,
    )


@pytest.mark.parametrize("device_type", AVAILABLE_DEVICES, ids=AVAILABLE_DEVICES_IDS)
def test_task_2_mask_only_different_dim(device_type: str):
    device = make_device(device_type)
    user_output = causal_mask(3, 5, torch.float32, device)
    assert_allclose(
        user_output,
        torch.tensor(
            [
                [0, 0, 0, -torch.inf, -torch.inf],
                [0, 0, 0, 0, -torch.inf],
                [0, 0, 0, 0, 0],
            ],
            device=device,
        ),
        precision=torch.float32,
    )


@pytest.mark.parametrize("device_type", AVAILABLE_DEVICES, ids=AVAILABLE_DEVICES_IDS)
@pytest.mark.parametrize("precision", PRECISIONS, ids=PRECISION_IDS)
@pytest.mark.parametrize(
    "batch_dimension", [0, 1, 2], ids=["batch_0", "batch_1", "batch_2"]
)
@pytest.mark.parametrize("scale", [None, 0.8])
def test_task_2_grouped_attention_causal_mask(
    device_type: str,
    precision: torch.dtype,
    batch_dimension: int,
    scale: float | None,
):
    grouped_attention_helper(device_type, precision, batch_dimension, scale, True)


def reference_qwen2_attention(
    x: torch.Tensor,
    wq: torch.Tensor,
    wk: torch.Tensor,
    wv: torch.Tensor,
    wo: torch.Tensor,
    bq: torch.Tensor,
    bk: torch.Tensor,
    bv: torch.Tensor,
    num_heads: int,
    num_kv_heads: int,
    max_seq_len: int,
    theta: int,
    mask: torch.Tensor | str | None,
) -> torch.Tensor:
    batch_size, seq_len, hidden_size = x.shape
    head_dim = hidden_size // num_heads
    scale = 1.0 / math.sqrt(head_dim)

    rope = RoPE(
        head_dim,
        max_seq_len,
        theta,
        traditional=False,
        device=x.device,
    )
    projection_q = linear(x, wq, bias=bq).reshape(batch_size, seq_len, num_heads, head_dim)
    projection_k = linear(x, wk, bias=bk).reshape(
        batch_size, seq_len, num_kv_heads, head_dim
    )
    projection_v = linear(x, wv, bias=bv).reshape(
        batch_size, seq_len, num_kv_heads, head_dim
    )
    projection_q = rope(projection_q, offset=slice(0, seq_len)).transpose(1, 2)
    projection_k = rope(projection_k, offset=slice(0, seq_len)).transpose(1, 2)
    projection_v = projection_v.transpose(1, 2)
    output = scaled_dot_product_attention_grouped(
        projection_q.to(dtype=torch.float32),
        projection_k.to(dtype=torch.float32),
        projection_v.to(dtype=torch.float32),
        scale=scale,
        mask=mask,
    ).to(dtype=x.dtype)
    output = output.transpose(1, 2).reshape(batch_size, seq_len, hidden_size)
    return linear(output, wo)


@pytest.mark.parametrize("device_type", AVAILABLE_DEVICES, ids=AVAILABLE_DEVICES_IDS)
@pytest.mark.parametrize("precision", PRECISIONS, ids=PRECISION_IDS)
@pytest.mark.parametrize("mask", [None, "causal"], ids=["no_mask", "causal_mask"])
def test_task_3_qwen2_grouped_query_attention(
    device_type: str,
    precision: torch.dtype,
    mask: str | None,
):
    device = make_device(device_type)
    batch_size = 1
    seq_len = 4
    hidden_size = 32
    num_heads = 4
    num_kv_heads = 2
    max_seq_len = 64
    theta = 10000
    head_dim = hidden_size // num_heads

    for _ in range(100):
        x = rand_uniform((batch_size, seq_len, hidden_size), device, precision)
        wq = rand_uniform((num_heads * head_dim, hidden_size), device, precision)
        wk = rand_uniform((num_kv_heads * head_dim, hidden_size), device, precision)
        wv = rand_uniform((num_kv_heads * head_dim, hidden_size), device, precision)
        wo = rand_uniform((hidden_size, num_heads * head_dim), device, precision)
        bq = rand_uniform((num_heads * head_dim,), device, precision)
        bk = rand_uniform((num_kv_heads * head_dim,), device, precision)
        bv = rand_uniform((num_kv_heads * head_dim,), device, precision)

        user_attention = Qwen2MultiHeadAttention(
            hidden_size=hidden_size,
            num_heads=num_heads,
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
        user_output = user_attention(x, mask=mask)

        reference_output = reference_qwen2_attention(
            x=x,
            wq=wq,
            wk=wk,
            wv=wv,
            wo=wo,
            bq=bq,
            bk=bk,
            bv=bv,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            max_seq_len=max_seq_len,
            theta=theta,
            mask=mask,
        )

        assert_allclose(
            user_output,
            reference_output,
            precision=precision,
            atol=5e-6 if precision == torch.float32 else 1e-3,
        )
