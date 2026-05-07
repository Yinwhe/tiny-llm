import pytest
import torch

from .tiny_llm_base import RoPE
from .utils import AVAILABLE_DEVICES, AVAILABLE_DEVICES_IDS, PRECISION_IDS, PRECISIONS, make_device, rand_uniform


@pytest.mark.parametrize("device_type", AVAILABLE_DEVICES, ids=AVAILABLE_DEVICES_IDS)
@pytest.mark.parametrize("traditional", [True, False])
@pytest.mark.parametrize("precision", PRECISIONS, ids=PRECISION_IDS)
def test_rope_week2_batch_offset(
    device_type: str, traditional: bool, precision: torch.dtype
):
    batch_size = 1
    num_heads = 8
    head_dim = 4
    max_seq_len = 20
    seq_len = 10
    base = 10000

    device = make_device(device_type)
    for _ in range(100):
        user_layer = RoPE(head_dim, max_seq_len, base, traditional=traditional, device=device)
        x = rand_uniform((batch_size, seq_len, num_heads, head_dim), device=device, precision=precision)
        input_pos_user = [slice(i, i + seq_len) for i in range(batch_size)]
        output = user_layer(x, input_pos_user)
        assert output.shape == x.shape
        assert output.dtype == x.dtype

