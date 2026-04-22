import numpy as np
import pytest
import torch

from .tiny_llm_base import RoPE
from .utils import (
    AVAILABLE_DEVICES,
    AVAILABLE_DEVICES_IDS,
    PRECISION_IDS,
    PRECISIONS,
    assert_allclose,
    make_device,
    rand_uniform,
)


def reference_rope(
    x: torch.Tensor,
    seq_len: int,
    base: int,
    traditional: bool,
    offset: slice | None,
) -> torch.Tensor:
    n, s, h, d = x.shape
    assert d % 2 == 0, "dims must be even"
    half_dims = d // 2

    inner = torch.arange(0, half_dims, device=x.device, dtype=torch.float32) / half_dims
    freqs = torch.pow(
        torch.tensor(base, device=x.device, dtype=torch.float32),
        -inner,
    )
    positions = torch.arange(seq_len, device=x.device, dtype=torch.float32)
    freqs = torch.outer(positions, freqs)
    cos_freqs = torch.cos(freqs)
    sin_freqs = torch.sin(freqs)

    cos_basis = cos_freqs[:s, :] if offset is None else cos_freqs[offset, :]
    sin_basis = sin_freqs[:s, :] if offset is None else sin_freqs[offset, :]
    cos_basis = cos_basis.reshape(-1, s, 1, half_dims)
    sin_basis = sin_basis.reshape(-1, s, 1, half_dims)

    if traditional:
        x_reshaped = x.reshape(n, s, h, half_dims, 2)
        x1 = x_reshaped[..., 0]
        x2 = x_reshaped[..., 1]
    else:
        x1 = x[..., :half_dims]
        x2 = x[..., half_dims:d]

    real = x1 * cos_basis - x2 * sin_basis
    imag = x2 * cos_basis + x1 * sin_basis

    if traditional:
        output = torch.stack([real, imag], dim=-1).reshape(n, s, h, d)
    else:
        output = torch.cat([real, imag], dim=-1).reshape(n, s, h, d)
    return output.to(dtype=x.dtype)


def rope_helper(
    device_type: str,
    traditional: bool,
    precision: torch.dtype,
    with_offset: bool,
):
    batch_size = 1
    num_heads = 8
    head_dim = 4
    max_seq_len = 20
    seq_len = 10
    base = 10000
    device = make_device(device_type)

    for _ in range(100):
        user_layer = RoPE(
            head_dim,
            max_seq_len,
            base,
            traditional=traditional,
            device=device,
        )
        x = rand_uniform((batch_size, seq_len, num_heads, head_dim), device, precision)

        if with_offset:
            input_pos = np.random.randint(0, max_seq_len - seq_len)
            input_pos_user = slice(input_pos, input_pos + seq_len)
        else:
            input_pos_user = None

        reference_output = reference_rope(
            x=x,
            seq_len=max_seq_len,
            base=base,
            traditional=traditional,
            offset=input_pos_user,
        )
        user_output = user_layer(x, input_pos_user)
        assert_allclose(
            user_output,
            reference_output,
            precision,
            atol=5e-6 if precision == torch.float32 else 1e-3,
        )


@pytest.mark.parametrize("device_type", AVAILABLE_DEVICES, ids=AVAILABLE_DEVICES_IDS)
@pytest.mark.parametrize(
    "with_offset", [True, False], ids=["with_offset", "without_offset"]
)
@pytest.mark.parametrize("precision", PRECISIONS, ids=PRECISION_IDS)
def test_task_1_rope_traditional(
    device_type: str, with_offset: bool, precision: torch.dtype
):
    rope_helper(device_type, True, precision, with_offset)


@pytest.mark.parametrize("device_type", AVAILABLE_DEVICES, ids=AVAILABLE_DEVICES_IDS)
@pytest.mark.parametrize(
    "with_offset", [True, False], ids=["with_offset", "without_offset"]
)
@pytest.mark.parametrize("precision", PRECISIONS, ids=PRECISION_IDS)
def test_task_2_rope_non_traditional(
    device_type: str, with_offset: bool, precision: torch.dtype
):
    rope_helper(device_type, False, precision, with_offset)
