import pytest
import torch
import torch.nn.functional as F

from .tiny_llm_base import Qwen2MLP, RMSNorm, linear, silu
from .utils import (
    AVAILABLE_DEVICES,
    AVAILABLE_DEVICES_IDS,
    PRECISION_IDS,
    PRECISIONS,
    assert_allclose,
    make_device,
    rand_uniform,
)


def reference_rms_norm(
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    original_dtype = x.dtype
    x = x.to(dtype=torch.float32)
    weight = weight.to(dtype=torch.float32)
    output = weight * x * torch.rsqrt(torch.mean(torch.square(x), dim=-1, keepdim=True) + eps)
    return output.to(dtype=original_dtype)


@pytest.mark.parametrize("device_type", AVAILABLE_DEVICES, ids=AVAILABLE_DEVICES_IDS)
@pytest.mark.parametrize("precision", PRECISIONS, ids=PRECISION_IDS)
def test_task_1_rms_norm(device_type: str, precision: torch.dtype):
    device = make_device(device_type)
    size = 100
    size_y = 111

    for _ in range(100):
        data = rand_uniform((size, size_y), device, precision)
        weight = rand_uniform((size_y,), device, precision)
        eps = torch.finfo(precision).eps
        reference_output = reference_rms_norm(data, weight, eps)
        user_output = RMSNorm(size_y, weight, eps=eps)(data)
        assert_allclose(user_output, reference_output, precision)


@pytest.mark.parametrize("device_type", AVAILABLE_DEVICES, ids=AVAILABLE_DEVICES_IDS)
def test_task_1_rms_norm_cast_to_float32(device_type: str):
    device = make_device(device_type)
    precision = torch.float16
    size = 32
    size_y = 64

    data = (torch.rand((size, size_y), device=device, dtype=precision) * 2000) - 1000
    weight = (torch.rand((size_y,), device=device, dtype=precision) * 2000) - 1000
    eps = torch.finfo(precision).eps

    user_output = RMSNorm(size_y, weight, eps=eps)(data)
    reference_output = reference_rms_norm(data, weight, eps)

    assert_allclose(user_output, reference_output, precision)


@pytest.mark.parametrize("device_type", AVAILABLE_DEVICES, ids=AVAILABLE_DEVICES_IDS)
@pytest.mark.parametrize("precision", PRECISIONS, ids=PRECISION_IDS)
def test_task_2_silu(device_type: str, precision: torch.dtype):
    device = make_device(device_type)
    batch_size = 10
    dim = 10

    for _ in range(100):
        x = rand_uniform((batch_size, dim), device, precision)
        user_output = silu(x)
        reference_output = F.silu(x)
        assert_allclose(user_output, reference_output, precision=precision)


DIM_PARAMS = [
    {"batch_size": 1, "seq_len": 5, "dim": 4, "hidden_dim": 8, "id": "small_dims"},
    {"batch_size": 2, "seq_len": 16, "dim": 32, "hidden_dim": 64, "id": "large_dims"},
    {
        "batch_size": 1,
        "seq_len": 1,
        "dim": 128,
        "hidden_dim": 256,
        "id": "single_token",
    },
]
DIM_PARAMS_IDS = [d["id"] for d in DIM_PARAMS]


def reference_qwen_mlp(
    x: torch.Tensor,
    w_gate: torch.Tensor,
    w_up: torch.Tensor,
    w_down: torch.Tensor,
) -> torch.Tensor:
    return linear(F.silu(linear(x, w_gate)) * linear(x, w_up), w_down)


@pytest.mark.parametrize("device_type", AVAILABLE_DEVICES, ids=AVAILABLE_DEVICES_IDS)
@pytest.mark.parametrize("precision", PRECISIONS, ids=PRECISION_IDS)
@pytest.mark.parametrize("dims", DIM_PARAMS, ids=DIM_PARAMS_IDS)
def test_task_2_qwen_mlp(
    device_type: str,
    precision: torch.dtype,
    dims: dict[str, int | str],
):
    device = make_device(device_type)
    batch_size = int(dims["batch_size"])
    seq_len = int(dims["seq_len"])
    dim = int(dims["dim"])
    hidden_dim = int(dims["hidden_dim"])

    x = rand_uniform((batch_size, seq_len, dim), device, precision)
    w_gate = rand_uniform((hidden_dim, dim), device, precision)
    w_up = rand_uniform((hidden_dim, dim), device, precision)
    w_down = rand_uniform((dim, hidden_dim), device, precision)

    user_mlp = Qwen2MLP(
        dim=dim,
        hidden_dim=hidden_dim,
        w_gate=w_gate,
        w_up=w_up,
        w_down=w_down,
    )
    user_output = user_mlp(x)
    reference_output = reference_qwen_mlp(x, w_gate, w_up, w_down)

    assert_allclose(user_output, reference_output, precision)
