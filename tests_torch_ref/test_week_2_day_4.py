import time

import pytest
import torch

from .tiny_llm_base import flash_attention, scaled_dot_product_attention_grouped
from .utils import assert_allclose, make_device


def attention_helper(
    device: torch.device,
    H_q: int,
    H: int,
    L: int,
    E: int,
    S: int,
    batch_size: int,
    mask_mode: str,
):
    precision = torch.float32
    q_shape = (batch_size, H_q, L, E)
    kv_shape = (batch_size, H, S, E)
    mask_shape = (batch_size, H_q, L, S)
    scale = 0.9

    for _ in range(100):
        query = torch.rand(q_shape, device=device, dtype=precision)
        key = torch.rand(kv_shape, device=device, dtype=precision)
        value = torch.rand(kv_shape, device=device, dtype=precision)
        if mask_mode == "no_mask":
            mask = None
        elif mask_mode == "mask":
            mask = torch.rand(mask_shape, device=device, dtype=precision)
        elif mask_mode == "causal":
            mask = "causal"
        else:
            raise ValueError(f"Unknown mask_mode: {mask_mode}")

        reference_output = scaled_dot_product_attention_grouped(
            query,
            key,
            value,
            scale=scale,
            mask=mask,
        )
        user_output = flash_attention(
            query,
            key,
            value,
            scale=scale,
            mask=mask,
        )
        assert_allclose(user_output, reference_output, precision=torch.float16)


def time_flash_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    scale: float,
    mask: torch.Tensor | str,
    num_iters: int = 4,
) -> float:
    if query.is_cuda:
        torch.cuda.synchronize(query.device)
    start = time.perf_counter()
    for _ in range(num_iters):
        flash_attention(query, key, value, scale=scale, mask=mask)
    if query.is_cuda:
        torch.cuda.synchronize(query.device)
    return (time.perf_counter() - start) / num_iters


def median(values: list[float]) -> float:
    values = sorted(values)
    return values[len(values) // 2]


def assert_causal_mask_faster_than_all_zero_mask(
    device: torch.device,
    batch: int,
    h_q: int,
    h: int,
    l: int,
    s: int,
    e: int,
    scale: float = 0.9,
):
    precision = torch.float32
    q_shape = (batch, h_q, l, e)
    kv_shape = (batch, h, s, e)
    mask_shape = (batch, h_q, l, s)

    query = torch.rand(q_shape, device=device, dtype=precision)
    key = torch.rand(kv_shape, device=device, dtype=precision)
    value = torch.rand(kv_shape, device=device, dtype=precision)
    zero_mask = torch.zeros(mask_shape, device=device, dtype=precision)

    for _ in range(3):
        flash_attention(query, key, value, scale=scale, mask="causal")
        flash_attention(query, key, value, scale=scale, mask=zero_mask)
    if query.is_cuda:
        torch.cuda.synchronize(query.device)

    causal_samples = []
    zero_mask_samples = []
    for round_idx in range(6):
        if round_idx % 2 == 0:
            causal_samples.append(
                time_flash_attention(query, key, value, scale=scale, mask="causal")
            )
            zero_mask_samples.append(
                time_flash_attention(query, key, value, scale=scale, mask=zero_mask)
            )
        else:
            zero_mask_samples.append(
                time_flash_attention(query, key, value, scale=scale, mask=zero_mask)
            )
            causal_samples.append(
                time_flash_attention(query, key, value, scale=scale, mask="causal")
            )

    causal_time = median(causal_samples)
    zero_mask_time = median(zero_mask_samples)
    assert causal_time < zero_mask_time, (
        "Expected causal mask to be faster than an all-zero mask, got "
        f"causal={causal_time:.6f}s and zero_mask={zero_mask_time:.6f}s."
    )


@pytest.mark.parametrize("mask_mode", ["no_mask", "mask", "causal"])
def test_task_2_flash_attention_cpu_small(mask_mode: str):
    attention_helper(make_device("cpu"), 6, 3, 2, 5, 3, 1, mask_mode)


@pytest.mark.parametrize("mask_mode", ["no_mask", "mask"])
def test_task_2_flash_attention_cpu(mask_mode: str):
    attention_helper(make_device("cpu"), 18, 6, 7, 5, 3, 10, mask_mode)


@pytest.mark.parametrize("mask_mode", ["no_mask", "mask", "causal"])
def test_task_2_flash_attention_cpu_large(mask_mode: str):
    attention_helper(make_device("cpu"), 28, 4, 16, 128, 16, 3, mask_mode)


def test_task_2_flash_attention_cpu_causal_mask_faster_than_all_zero_mask():
    assert_causal_mask_faster_than_all_zero_mask(
        device=make_device("cpu"),
        batch=1,
        h_q=8,
        h=8,
        l=128,
        s=128,
        e=128,
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
@pytest.mark.parametrize("mask_mode", ["no_mask", "mask"])
def test_task_3_flash_attention_gpu_extra_small(mask_mode: str):
    attention_helper(make_device("cuda"), 1, 1, 5, 7, 4, 1, mask_mode)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
@pytest.mark.parametrize("mask_mode", ["no_mask", "mask", "causal"])
def test_task_3_flash_attention_gpu_small(mask_mode: str):
    attention_helper(make_device("cuda"), 6, 3, 2, 5, 3, 1, mask_mode)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
@pytest.mark.parametrize("mask_mode", ["no_mask", "mask"])
def test_task_3_flash_attention_gpu(mask_mode: str):
    attention_helper(make_device("cuda"), 18, 6, 7, 5, 3, 10, mask_mode)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
@pytest.mark.parametrize("mask_mode", ["no_mask", "mask", "causal"])
def test_task_3_flash_attention_gpu_large(mask_mode: str):
    attention_helper(make_device("cuda"), 28, 4, 16, 128, 16, 3, mask_mode)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_task_3_flash_attention_gpu_causal_mask_faster_than_all_zero_mask():
    assert_causal_mask_faster_than_all_zero_mask(
        device=make_device("cuda"),
        batch=2,
        h_q=8,
        h=8,
        l=512,
        s=512,
        e=128,
    )
