import torch
import pytest

from .tiny_llm_base import quantized_matmul
from .utils import assert_allclose, make_device


AWQ_PACK_ORDER = [0, 2, 4, 6, 1, 3, 5, 7]


def pack_awq_int4(logical_int4: torch.Tensor) -> torch.Tensor:
    """
    Pack a logical int4 matrix of shape [rows, cols] into AWQ's raw uint32
    storage layout, where every 8 consecutive columns are packed into one word.
    """
    rows, cols = logical_int4.shape
    if cols % 8 != 0:
        raise ValueError("pack_awq_int4 expects the last dimension to be divisible by 8")

    ordered = logical_int4.view(rows, cols // 8, 8)[:, :, AWQ_PACK_ORDER]
    shifts = torch.arange(0, 32, 4, device=logical_int4.device, dtype=torch.int32)
    packed = torch.bitwise_left_shift(ordered.to(torch.int32), shifts[None, None, :]).sum(dim=-1)
    return packed.to(torch.int32)


def dequantize_awq(
    qweight_logical: torch.Tensor,
    qzeros_logical: torch.Tensor,
    scales: torch.Tensor,
    group_size: int,
) -> torch.Tensor:
    zeros = qzeros_logical.to(torch.float32).repeat_interleave(group_size, dim=0)
    repeated_scales = scales.to(torch.float32).repeat_interleave(group_size, dim=0)
    return (qweight_logical.to(torch.float32) - zeros) * repeated_scales


def quantized_matmul_helper(
    device: torch.device, identity_matrix: bool, precision: torch.dtype
):
    if identity_matrix:
        a = torch.eye(64, device=device, dtype=precision)
    else:
        a = torch.randn(3, 64, device=device, dtype=precision)

    # AWQ packs 8 output columns into one uint32 word, so this toy shape uses K=8.
    n = 64
    k = 8
    group_size = 64
    scales = 0.05 + torch.rand(n // group_size, k, device=device, dtype=precision)
    qzeros_logical = torch.randint(0, 16, (n // group_size, k), device=device, dtype=torch.int32)
    qweight_logical = torch.randint(0, 16, (n, k), device=device, dtype=torch.int32)

    qzeros = pack_awq_int4(qzeros_logical)
    qweight = pack_awq_int4(qweight_logical)

    user_out = quantized_matmul(
        scales=scales,
        zeros=qzeros,
        group_size=group_size,
        bits=4,
        a=a,
        b=qweight,
        transpose_b=True,
    )

    dequantized_weight = dequantize_awq(qweight_logical, qzeros_logical, scales, group_size).to(precision)
    ref_out = a @ dequantized_weight
    assert_allclose(user_out, ref_out, precision=precision, atol=5e-2)


def test_task_2_quantized_matmul_simple_f16_cpu():
    quantized_matmul_helper(make_device("cpu"), True, torch.float16)


def test_task_2_quantized_matmul_complex_f16_cpu():
    quantized_matmul_helper(make_device("cpu"), False, torch.float16)


def test_task_2_quantized_matmul_simple_f32_cpu():
    quantized_matmul_helper(make_device("cpu"), True, torch.float32)


def test_task_2_quantized_matmul_complex_f32_cpu():
    quantized_matmul_helper(make_device("cpu"), False, torch.float32)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_task_3_quantized_matmul_simple_f16_gpu():
    quantized_matmul_helper(make_device("cuda"), True, torch.float16)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_task_3_quantized_matmul_complex_f16_gpu():
    quantized_matmul_helper(make_device("cuda"), False, torch.float16)
