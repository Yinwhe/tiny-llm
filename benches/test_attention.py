import math

import pytest
import torch
import tiny_llm_ref

from .utils import assert_allclose


def get_test_attention_data():
    device = torch.device("cuda")
    dtype = torch.float32

    q = torch.empty((10, 28, 1024, 128), device=device, dtype=dtype)
    k = torch.empty((10, 4, 1024, 128), device=device, dtype=dtype)
    v = torch.empty((10, 4, 1024, 128), device=device, dtype=dtype)
    torch.nn.init.kaiming_uniform_(q, a=math.sqrt(5))
    torch.nn.init.kaiming_uniform_(k, a=math.sqrt(5))
    torch.nn.init.kaiming_uniform_(v, a=math.sqrt(5))

    repeats = q.shape[1] // k.shape[1]
    k_expanded = k.repeat_interleave(repeats, dim=1)
    v_expanded = v.repeat_interleave(repeats, dim=1)
    res = torch.nn.functional.scaled_dot_product_attention(
        q,
        k_expanded,
        v_expanded,
        scale=1.0,
    )
    return q, k, v, res


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_torch_attention(benchmark):
    q, k, v, res = get_test_attention_data()
    repeats = q.shape[1] // k.shape[1]
    k_expanded = k.repeat_interleave(repeats, dim=1)
    v_expanded = v.repeat_interleave(repeats, dim=1)
    result = benchmark(
        lambda: torch.nn.functional.scaled_dot_product_attention(
            q,
            k_expanded,
            v_expanded,
            scale=1.0,
        )
    )
    assert_allclose(result, res, precision=torch.float32, rtol=1e-2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_refsol_attention(benchmark):
    q, k, v, res = get_test_attention_data()
    result = benchmark(
        lambda: tiny_llm_ref.scaled_dot_product_attention_grouped(
            q, k, v, scale=1.0
        )
    )
    assert_allclose(result, res, precision=torch.float32, rtol=1e-2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_refsol_flash_attention(benchmark):
    q, k, v, res = get_test_attention_data()
    result = benchmark(lambda: tiny_llm_ref.flash_attention(q, k, v, scale=1.0))
    assert_allclose(result, res, precision=torch.float32, rtol=1e-2)
