import pytest
import torch
import tiny_llm_ref

from .utils import assert_allclose


AWQ_PACK_ORDER = [0, 2, 4, 6, 1, 3, 5, 7]


def pack_awq_int4(logical_int4: torch.Tensor) -> torch.Tensor:
    rows, cols = logical_int4.shape
    if cols % 8 != 0:
        raise ValueError("pack_awq_int4 expects the last dimension to be divisible by 8")

    ordered = logical_int4.view(rows, cols // 8, 8)[:, :, AWQ_PACK_ORDER]
    shifts = torch.arange(0, 32, 4, device=logical_int4.device, dtype=torch.int32)
    packed = torch.bitwise_left_shift(
        ordered.to(torch.int32), shifts[None, None, :]
    ).sum(dim=-1)
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


def get_test_matmul_data():
    device = torch.device("cuda")
    dtype = torch.float16
    torch.manual_seed(0)

    in_features = 3584
    out_features = 512
    group_size = 64

    x = torch.randn((300, in_features), device=device, dtype=dtype)
    scales = 0.05 + torch.rand(
        (in_features // group_size, out_features),
        device=device,
        dtype=dtype,
    )
    qzeros_logical = torch.randint(
        0,
        16,
        (in_features // group_size, out_features),
        device=device,
        dtype=torch.int32,
    )
    qweight_logical = torch.randint(
        0,
        16,
        (in_features, out_features),
        device=device,
        dtype=torch.int32,
    )

    qzeros = pack_awq_int4(qzeros_logical)
    qweight = pack_awq_int4(qweight_logical)
    dequantized_weight = dequantize_awq(
        qweight_logical, qzeros_logical, scales, group_size
    ).to(dtype)
    res = x @ dequantized_weight
    return qweight, scales, qzeros, x, res


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_refsol_quantized_matmul(benchmark):
    qweight, scales, qzeros, x, res = get_test_matmul_data()
    result = benchmark(
        lambda: tiny_llm_ref.quantized_matmul(
            scales, qzeros, 64, 4, x, qweight, transpose_b=True
        )
    )
    assert_allclose(result, res, precision=torch.float16, rtol=5e-2, atol=2e-1)
