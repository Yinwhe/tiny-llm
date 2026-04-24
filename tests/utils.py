import torch

AVAILABLE_DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
AVAILABLE_DEVICES_IDS = AVAILABLE_DEVICES
PRECISIONS = [torch.float32, torch.float16]
PRECISION_IDS = ["f32", "f16"]


def assert_allclose(
    a: torch.Tensor,
    b: torch.Tensor,
    precision: torch.dtype,
    rtol: float | None = None,
    atol: float | None = None,
    message: str | None = None,
):
    if precision == torch.float32:
        rtol = rtol or 1.0e-5
        atol = atol or 1.0e-6
    elif precision == torch.float16:
        rtol = rtol or 3.0e-2
        atol = atol or 1.0e-5
    else:
        raise ValueError(f"Unsupported precision: {precision}")
    torch.testing.assert_close(a, b, rtol=rtol, atol=atol, msg=message)


def make_device(device_type: str) -> torch.device:
    return torch.device(device_type)


def rand_uniform(
    shape: tuple[int, ...],
    device: torch.device,
    precision: torch.dtype,
) -> torch.Tensor:
    return torch.rand(shape, device=device, dtype=precision)
