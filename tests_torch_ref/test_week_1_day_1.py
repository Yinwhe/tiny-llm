import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from .tiny_llm_base import *
from .utils import *


@pytest.mark.parametrize("device_type", AVAILABLE_DEVICES, ids=AVAILABLE_DEVICES_IDS)
@pytest.mark.parametrize("precision", PRECISIONS, ids=PRECISION_IDS)
def test_task_1_softmax(device_type: str, precision: torch.dtype):
    device = make_device(device_type)
    batch_size = 10
    dim = 10
    for _ in range(100):
        x = rand_uniform((batch_size, dim), device=device, precision=precision)
        user_output = softmax(x, axis=-1)
        reference_output = torch.softmax(x, dim=-1)
        assert_allclose(user_output, reference_output, precision=precision)


@pytest.mark.parametrize("device_type", AVAILABLE_DEVICES, ids=AVAILABLE_DEVICES_IDS)
@pytest.mark.parametrize("precision", PRECISIONS, ids=PRECISION_IDS)
@pytest.mark.parametrize(
    "batch_dimension", [0, 1, 2], ids=["batch_0", "batch_1", "batch_2"]
)
def test_task_1_simple_attention(
    device_type: str, precision: torch.dtype, batch_dimension: int
):
    """
    Test if `scaled_dot_product_attention_simple` can process Q/K/V correctly.
    We assume Q/K/V are of the same dimensions and test different batch dimensions.
    """
    device = make_device(device_type)
    if batch_dimension == 0:
        batch_size = ()
    elif batch_dimension == 1:
        batch_size = (2, 3)
    elif batch_dimension == 2:
        batch_size = (2, 3, 3)
    else:
        raise ValueError(f"Unsupported batch dimension: {batch_dimension}")
    dim_l = 4
    dim_d = 5
    for _ in range(100):
        query = rand_uniform((*batch_size, dim_l, dim_d), device, precision)
        key = rand_uniform((*batch_size, dim_l, dim_d), device, precision)
        value = rand_uniform((*batch_size, dim_l, dim_d), device, precision)
        reference_output = F.scaled_dot_product_attention(
            query=query.reshape(1, -1, dim_l, dim_d),
            key=key.reshape(1, -1, dim_l, dim_d),
            value=value.reshape(1, -1, dim_l, dim_d),
            scale=1.0 / (dim_d**0.5),
        ).reshape(*batch_size, dim_l, dim_d)
        user_output = scaled_dot_product_attention_simple(
            query,
            key,
            value,
        )
        assert_allclose(user_output, reference_output, precision=precision)


@pytest.mark.parametrize("device_type", AVAILABLE_DEVICES, ids=AVAILABLE_DEVICES_IDS)
@pytest.mark.parametrize("precision", PRECISIONS, ids=PRECISION_IDS)
@pytest.mark.parametrize(
    "batch_dimension", [0, 1, 2], ids=["batch_0", "batch_1", "batch_2"]
)
def test_task_1_simple_attention_scale_mask(
    device_type: str, precision: torch.dtype, batch_dimension: int
):
    """
    Test if `scaled_dot_product_attention_simple` can process scale and mask correctly.
    """
    device = make_device(device_type)
    if batch_dimension == 0:
        batch_size = ()
    elif batch_dimension == 1:
        batch_size = (2, 3)
    elif batch_dimension == 2:
        batch_size = (2, 3, 3)
    else:
        raise ValueError(f"Unsupported batch dimension: {batch_dimension}")
    dim_l = 4
    dim_d = 5
    for _ in range(100):
        query = rand_uniform((*batch_size, dim_l, dim_d), device, precision)
        key = rand_uniform((*batch_size, dim_l, dim_d), device, precision)
        value = rand_uniform((*batch_size, dim_l, dim_d), device, precision)
        mask = rand_uniform((*batch_size, dim_l, dim_l), device, precision)
        scale = 0.5
        reference_output = F.scaled_dot_product_attention(
            query=query.reshape(1, -1, dim_l, dim_d),
            key=key.reshape(1, -1, dim_l, dim_d),
            value=value.reshape(1, -1, dim_l, dim_d),
            attn_mask=mask.reshape(1, -1, dim_l, dim_l),
            scale=scale,
        ).reshape(*batch_size, dim_l, dim_d)
        user_output = scaled_dot_product_attention_simple(
            query,
            key,
            value,
            scale=scale,
            mask=mask,
        )
        assert_allclose(user_output, reference_output, precision=precision)


@pytest.mark.parametrize("device_type", AVAILABLE_DEVICES, ids=AVAILABLE_DEVICES_IDS)
@pytest.mark.parametrize("precision", PRECISIONS, ids=PRECISION_IDS)
def test_task_2_linear(device_type: str, precision: torch.dtype):
    device = make_device(device_type)
    batch_size = 10
    dim_y = 10
    dim_x = 12
    for _ in range(100):
        x = rand_uniform((batch_size, dim_x), device, precision)
        w = rand_uniform((dim_y, dim_x), device, precision)
        b = rand_uniform((dim_y,), device, precision)
        user_output = linear(x, w, b)
        if precision == torch.float16 and device.type == "cpu":
            break
        reference_output = F.linear(x, w, b)
        assert_allclose(user_output, reference_output, precision=precision)


@pytest.mark.parametrize("device_type", AVAILABLE_DEVICES, ids=AVAILABLE_DEVICES_IDS)
@pytest.mark.parametrize("precision", PRECISIONS, ids=PRECISION_IDS)
def test_task_2_simple_multi_head_attention(device_type: str, precision: torch.dtype):
    """
    Test if `MultiHeadAttention` can process everything correctly. We assume Q/K/V are of the same dimensions.
    """
    device = make_device(device_type)
    dim_l = 11
    dim_d = 9
    num_heads = 3
    batch_size = 10
    for _ in range(100):
        query = rand_uniform((batch_size, dim_l, num_heads * dim_d), device, precision)
        key = rand_uniform((batch_size, dim_l, num_heads * dim_d), device, precision)
        value = rand_uniform((batch_size, dim_l, num_heads * dim_d), device, precision)
        q_proj_weight = rand_uniform(
            (num_heads * dim_d, num_heads * dim_d), device, precision
        )
        k_proj_weight = rand_uniform(
            (num_heads * dim_d, num_heads * dim_d), device, precision
        )
        v_proj_weight = rand_uniform(
            (num_heads * dim_d, num_heads * dim_d), device, precision
        )
        out_proj_weight = rand_uniform(
            (num_heads * dim_d, num_heads * dim_d), device, precision
        )
        mask = rand_uniform((dim_l, dim_l), device, precision)

        reference_mha = nn.MultiheadAttention(
            num_heads * dim_d,
            num_heads,
            bias=False,
            batch_first=True,
            device=device,
            dtype=precision,
        )

        with torch.no_grad():
            reference_mha.in_proj_weight.copy_(
                torch.cat([q_proj_weight, k_proj_weight, v_proj_weight], dim=0)
            )
            reference_mha.out_proj.weight.copy_(out_proj_weight)

        reference_output, _ = reference_mha(
            query, key, value, attn_mask=mask, need_weights=False
        )

        user_output = SimpleMultiHeadAttention(
            num_heads * dim_d,
            num_heads,
            q_proj_weight,
            k_proj_weight,
            v_proj_weight,
            out_proj_weight,
        )(
            query,
            key,
            value,
            mask=mask,
        )
        assert_allclose(user_output, reference_output, precision=precision)
