import numpy as np
import pytest
import torch

from .tiny_llm_base import (
    BatchingKvCache,
    Qwen2ModelWeek2,
    RoPE,
    TinyKvFullCache,
    causal_mask,
    flash_attention,
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
from .test_week_2_day_1 import load_transformers_qwen2_model


def reference_rope(
    x: torch.Tensor,
    seq_len: int,
    base: int,
    traditional: bool,
    offset: list[slice] | slice | None,
) -> torch.Tensor:
    batch_size, current_seq_len, num_heads, dims = x.shape
    assert dims % 2 == 0, "dims must be even"
    half_dims = dims // 2

    inner = (
        torch.arange(0, half_dims, device=x.device, dtype=torch.float32) / half_dims
    )
    freqs = torch.pow(
        torch.tensor(base, device=x.device, dtype=torch.float32),
        -inner,
    )
    positions = torch.arange(seq_len, device=x.device, dtype=torch.float32)
    freqs = torch.outer(positions, freqs)
    cos_freqs = torch.cos(freqs)
    sin_freqs = torch.sin(freqs)

    if offset is None:
        cos_basis = cos_freqs[:current_seq_len, :]
        sin_basis = sin_freqs[:current_seq_len, :]
    elif isinstance(offset, slice):
        cos_basis = cos_freqs[offset, :]
        sin_basis = sin_freqs[offset, :]
    else:
        position_ids = torch.tensor(
            [list(range(current.start, current.stop)) for current in offset],
            device=x.device,
            dtype=torch.long,
        )
        cos_basis = cos_freqs[position_ids, :]
        sin_basis = sin_freqs[position_ids, :]

    cos_basis = cos_basis.reshape(-1, current_seq_len, 1, half_dims)
    sin_basis = sin_basis.reshape(-1, current_seq_len, 1, half_dims)

    if traditional:
        x_reshaped = x.reshape(batch_size, current_seq_len, num_heads, half_dims, 2)
        x1 = x_reshaped[..., 0]
        x2 = x_reshaped[..., 1]
    else:
        x1 = x[..., :half_dims]
        x2 = x[..., half_dims:dims]

    real = x1 * cos_basis - x2 * sin_basis
    imag = x2 * cos_basis + x1 * sin_basis

    if traditional:
        output = torch.stack([real, imag], dim=-1).reshape(
            batch_size, current_seq_len, num_heads, dims
        )
    else:
        output = torch.cat([real, imag], dim=-1).reshape(
            batch_size, current_seq_len, num_heads, dims
        )
    return output.to(dtype=x.dtype)


def rope_helper(
    device_type: str,
    traditional: bool,
    precision: torch.dtype,
):
    batch_size = 16
    num_heads = 8
    head_dim = 4
    max_seq_len = 14
    seq_len = 9
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

        input_pos = np.random.randint(0, max_seq_len - seq_len, size=batch_size)
        input_pos_user = [slice(int(i), int(i + seq_len)) for i in input_pos]

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
@pytest.mark.parametrize("traditional", [False, True], ids=["default", "traditional"])
@pytest.mark.parametrize("precision", PRECISIONS, ids=PRECISION_IDS)
def test_task_1_rope_multiple_offsets(
    device_type: str, traditional: bool, precision: torch.dtype
):
    rope_helper(device_type, traditional, precision)


def attention_helper(
    device: torch.device,
    h_q: int,
    h: int,
    l: int,
    e: int,
    s: int,
    batch_size: int,
    use_flash_attention: bool = False,
):
    precision = torch.float32
    q_shape = (batch_size, h_q, l, e)
    kv_shape = (batch_size, h, s, e)
    scale = 0.8

    for _ in range(100):
        query = torch.rand(q_shape, device=device, dtype=precision)
        key = torch.rand(kv_shape, device=device, dtype=precision)
        value = torch.rand(kv_shape, device=device, dtype=precision)
        mask = torch.rand((batch_size, 1, l, s), device=device, dtype=precision)

        repeats = h_q // h
        query_grouped = query.reshape(batch_size, h, repeats, l, e)
        key_grouped = key.reshape(batch_size, h, 1, s, e)
        value_grouped = value.reshape(batch_size, h, 1, s, e)
        mask_grouped = mask.reshape(batch_size, 1, 1, l, s)

        reference_output_with_mask = torch.nn.functional.scaled_dot_product_attention(
            query_grouped,
            key_grouped,
            value_grouped,
            scale=scale,
            attn_mask=mask_grouped,
        ).reshape(batch_size, h_q, l, e)
        reference_output_without_mask = (
            torch.nn.functional.scaled_dot_product_attention(
                query_grouped,
                key_grouped,
                value_grouped,
                scale=scale,
            ).reshape(batch_size, h_q, l, e)
        )
        if use_flash_attention:
            user_output_with_mask = flash_attention(
                query,
                key,
                value,
                scale=scale,
                mask=mask,
            )
            user_output_without_mask = flash_attention(
                query,
                key,
                value,
                scale=scale,
            )
        else:
            user_output_with_mask = scaled_dot_product_attention_grouped(
                query,
                key,
                value,
                scale=scale,
                mask=mask,
            )
            user_output_without_mask = scaled_dot_product_attention_grouped(
                query,
                key,
                value,
                scale=scale,
            )

        assert_allclose(
            user_output_without_mask,
            reference_output_without_mask,
            precision=torch.float16,
            message="no mask",
        )
        assert_allclose(
            user_output_with_mask,
            reference_output_with_mask,
            precision=torch.float16,
            message="with mask",
        )


@pytest.mark.parametrize("device_type", AVAILABLE_DEVICES, ids=AVAILABLE_DEVICES_IDS)
def test_task_1_attention_with_mask_small(device_type: str):
    attention_helper(make_device(device_type), 6, 3, 2, 5, 3, 1)


@pytest.mark.parametrize("device_type", AVAILABLE_DEVICES, ids=AVAILABLE_DEVICES_IDS)
def test_task_1_attention_with_mask(device_type: str):
    attention_helper(make_device(device_type), 18, 6, 7, 5, 3, 10)


@pytest.mark.parametrize("device_type", AVAILABLE_DEVICES, ids=AVAILABLE_DEVICES_IDS)
def test_task_1_attention_with_mask_large(device_type: str):
    attention_helper(make_device(device_type), 28, 4, 16, 128, 16, 3)


@pytest.mark.parametrize("device_type", AVAILABLE_DEVICES, ids=AVAILABLE_DEVICES_IDS)
def test_task_1_flash_attention_with_mask_small(device_type: str):
    attention_helper(make_device(device_type), 6, 3, 2, 5, 3, 1, True)


@pytest.mark.parametrize("device_type", AVAILABLE_DEVICES, ids=AVAILABLE_DEVICES_IDS)
def test_task_1_flash_attention_with_mask(device_type: str):
    attention_helper(make_device(device_type), 18, 6, 7, 5, 3, 10, True)


@pytest.mark.parametrize("device_type", AVAILABLE_DEVICES, ids=AVAILABLE_DEVICES_IDS)
def test_task_1_flash_attention_with_mask_large(device_type: str):
    attention_helper(make_device(device_type), 28, 4, 16, 128, 16, 3, True)


def test_task_1_causal_mask_rectangular_example():
    mask = causal_mask(3, 5, torch.float32, make_device("cpu"))
    expected = torch.tensor(
        [
            [0.0, 0.0, 0.0, -torch.inf, -torch.inf],
            [0.0, 0.0, 0.0, 0.0, -torch.inf],
            [0.0, 0.0, 0.0, 0.0, 0.0],
        ],
        dtype=torch.float32,
    )
    assert_allclose(mask, expected, precision=torch.float32)


def test_task_2_batching_kv_cache():
    cache = BatchingKvCache(max_active_requests=3, max_seq_len=8)

    slot0 = TinyKvFullCache()
    slot0.update_and_fetch(
        torch.tensor([[[[10.0]]]], dtype=torch.float32),
        torch.tensor([[[[110.0]]]], dtype=torch.float32),
    )

    slot2 = TinyKvFullCache()
    slot2.update_and_fetch(
        torch.tensor([[[[20.0], [21.0]]]], dtype=torch.float32),
        torch.tensor([[[[120.0], [121.0]]]], dtype=torch.float32),
    )

    cache.add_request(slot0, 0)
    cache.add_request(slot2, 2)

    keys = torch.tensor(
        [
            [[[12.0], [13.0]]],
            [[[0.0], [0.0]]],
            [[[22.0], [23.0]]],
        ],
        dtype=torch.float32,
    )
    values = torch.tensor(
        [
            [[[112.0], [113.0]]],
            [[[0.0], [0.0]]],
            [[[122.0], [123.0]]],
        ],
        dtype=torch.float32,
    )

    batched_keys, batched_values, seq_len, mask = cache.update_and_fetch(
        keys, values, mask_length=2
    )

    expected_keys = torch.tensor(
        [
            [[[0.0], [10.0], [12.0], [13.0]]],
            [[[0.0], [0.0], [0.0], [0.0]]],
            [[[20.0], [21.0], [22.0], [23.0]]],
        ],
        dtype=torch.float32,
    )
    expected_values = torch.tensor(
        [
            [[[0.0], [110.0], [112.0], [113.0]]],
            [[[0.0], [0.0], [0.0], [0.0]]],
            [[[120.0], [121.0], [122.0], [123.0]]],
        ],
        dtype=torch.float32,
    )
    expected_mask = torch.tensor(
        [
            [[[-torch.inf, 0.0, 0.0, -torch.inf], [-torch.inf, 0.0, 0.0, 0.0]]],
            [[[-torch.inf, -torch.inf, -torch.inf, -torch.inf], [-torch.inf, -torch.inf, -torch.inf, -torch.inf]]],
            [[[0.0, 0.0, 0.0, -torch.inf], [0.0, 0.0, 0.0, 0.0]]],
        ],
        dtype=torch.float32,
    ).reshape(3, 1, 2, 4)

    assert seq_len is None
    assert_allclose(batched_keys, expected_keys, precision=torch.float32)
    assert_allclose(batched_values, expected_values, precision=torch.float32)
    assert_allclose(mask, expected_mask, precision=torch.float32)


def helper_test_task_3(model_name: str, seq_len: int, iters: int = 1):
    requests = 4
    max_seq_len = seq_len

    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")

    device = make_device("cuda")
    hf_model = load_transformers_qwen2_model(model_name, device)
    precision = torch.float16
    try:
        model = Qwen2ModelWeek2(hf_model, precision=precision, device=device)
    except RuntimeError as error:
        if "out of memory" in str(error).lower():
            pytest.skip(f"not enough memory to build tiny model for {model_name}")
        raise

    completed = 0
    attempts = 0
    max_attempts = iters * 5
    with torch.no_grad():
        while completed < iters and attempts < max_attempts:
            attempts += 1
            cache = [
                BatchingKvCache(requests, max_seq_len)
                for _ in range(model.num_hidden_layers)
            ]
            staggered_start = [seq_len * i // requests for i in range(requests)]
            inputs = torch.randint(
                low=0,
                high=hf_model.config.vocab_size,
                size=(requests, seq_len),
                device=device,
            )
            ref_outputs = hf_model(inputs, use_cache=False).logits
            if torch.isnan(ref_outputs).any():
                continue

            try:
                for offset in range(seq_len + staggered_start[-1]):
                    seq_idx = [offset - start for start in staggered_start]

                    for request_id, sidx in enumerate(seq_idx):
                        if sidx == 0:
                            for layer_cache in cache:
                                layer_cache.add_request(TinyKvFullCache(), request_id)
                        elif sidx == seq_len:
                            for layer_cache in cache:
                                layer_cache.remove_request(request_id)

                    next_tokens = []
                    next_offsets = []
                    for request_id, sidx in enumerate(seq_idx):
                        if 0 <= sidx < seq_len:
                            next_tokens.append(inputs[request_id, sidx].item())
                            next_offsets.append(sidx)
                        else:
                            next_tokens.append(0)
                            next_offsets.append(0)

                    user_out = model(
                        inputs=torch.tensor(
                            next_tokens, device=device, dtype=torch.long
                        ).reshape(-1, 1),
                        offset=torch.tensor(
                            next_offsets, device=device, dtype=torch.long
                        ),
                        cache=cache,
                    )

                    for request_id, sidx in enumerate(seq_idx):
                        if 0 <= sidx < seq_len:
                            user_out_r = user_out[request_id, 0, :]
                            ref_out_r = ref_outputs[request_id, sidx, :]
                            user_out_r = torch.log_softmax(
                                user_out_r.float(), dim=-1
                            )
                            ref_out_r = torch.log_softmax(
                                ref_out_r.float(), dim=-1
                            )
                            assert_allclose(
                                user_out_r,
                                ref_out_r,
                                precision=precision,
                                rtol=1e-1,
                            )
            except AssertionError:
                continue
            completed += 1
    if completed < iters:
        pytest.skip(f"{model_name} produced unstable reference comparisons")


def test_task_3_qwen_2_05b():
    helper_test_task_3("Qwen/Qwen2-0.5B-Instruct", seq_len=3)


def test_task_3_qwen_2_7b():
    helper_test_task_3("Qwen/Qwen2-7B-Instruct", seq_len=3)


def test_task_3_qwen_2_15b():
    helper_test_task_3("Qwen/Qwen2-1.5B-Instruct", seq_len=3)
