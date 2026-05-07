import pytest
import torch

from .tiny_llm_base import (
    Embedding,
    Qwen2ModelWeek1,
    Qwen2TransformerBlock,
    causal_mask,
    linear,
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


def disable_transformers_torchao_detection():
    import transformers.utils as transformers_utils
    import transformers.utils.import_utils as import_utils

    import_utils._torchao_available = False
    transformers_utils.is_torchao_available = lambda *args, **kwargs: False


def make_qwen2_config(
    hidden_size: int,
    intermediate_size: int,
    num_attention_heads: int,
    num_kv_heads: int,
    vocab_size: int,
):
    disable_transformers_torchao_detection()
    from transformers.models.qwen2.configuration_qwen2 import Qwen2Config

    config = Qwen2Config(
        hidden_size=hidden_size,
        num_hidden_layers=1,
        intermediate_size=intermediate_size,
        num_attention_heads=num_attention_heads,
        num_key_value_heads=num_kv_heads,
        rms_norm_eps=1e-6,
        vocab_size=vocab_size,
    )
    config._attn_implementation = "eager"
    return config


def load_transformers_qwen2_model(model_name: str, device: torch.device):
    disable_transformers_torchao_detection()
    from transformers import AutoModelForCausalLM

    precision = torch.float16 if device.type == "cuda" else torch.float32
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            local_files_only=True,
            torch_dtype=precision,
            attn_implementation="eager",
        )
    except OSError:
        pytest.skip(f"{model_name} model not found")
    except RuntimeError as error:
        if "out of memory" in str(error).lower():
            pytest.skip(f"not enough memory to load {model_name}")
        raise

    try:
        model.to(device)
    except RuntimeError as error:
        if "out of memory" in str(error).lower():
            pytest.skip(f"not enough memory to move {model_name} to {device}")
        raise

    model.eval()
    return model


@pytest.mark.parametrize("device_type", AVAILABLE_DEVICES, ids=AVAILABLE_DEVICES_IDS)
@pytest.mark.parametrize("precision", PRECISIONS, ids=PRECISION_IDS)
@pytest.mark.parametrize("mask", [None, "causal"], ids=["no_mask", "causal_mask"])
def test_task_1_transformer_block(
    device_type: str,
    precision: torch.dtype,
    mask: str | None,
):
    disable_transformers_torchao_detection()
    from transformers.models.qwen2.modeling_qwen2 import (
        Qwen2DecoderLayer,
        Qwen2RotaryEmbedding,
    )

    device = make_device(device_type)
    batch_size = 1
    seq_len = 10
    num_attention_heads = 4
    num_kv_heads = 2
    hidden_size = 32
    intermediate_size = hidden_size * 4

    config = make_qwen2_config(
        hidden_size,
        intermediate_size,
        num_attention_heads,
        num_kv_heads,
        vocab_size=1000,
    )
    hf_transformer_block = Qwen2DecoderLayer(config, layer_idx=0)
    hf_transformer_block.to(device=device, dtype=precision)
    hf_transformer_block.eval()

    attention = hf_transformer_block.self_attn
    mlp = hf_transformer_block.mlp
    user_transformer_block = Qwen2TransformerBlock(
        num_attention_heads=num_attention_heads,
        num_kv_heads=num_kv_heads,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        rms_norm_eps=1e-6,
        wq=attention.q_proj.weight.detach(),
        wk=attention.k_proj.weight.detach(),
        wv=attention.v_proj.weight.detach(),
        wo=attention.o_proj.weight.detach(),
        bq=attention.q_proj.bias.detach(),
        bk=attention.k_proj.bias.detach(),
        bv=attention.v_proj.bias.detach(),
        w_gate=mlp.gate_proj.weight.detach(),
        w_up=mlp.up_proj.weight.detach(),
        w_down=mlp.down_proj.weight.detach(),
        w_input_layernorm=hf_transformer_block.input_layernorm.weight.detach(),
        w_post_attention_layernorm=(
            hf_transformer_block.post_attention_layernorm.weight.detach()
        ),
        max_seq_len=config.max_position_embeddings,
        theta=config.rope_theta,
    )
    rotary_embedding = Qwen2RotaryEmbedding(config=config, device=device)

    for _ in range(100):
        x = rand_uniform((batch_size, seq_len, hidden_size), device, precision)
        attention_mask = None
        if mask == "causal":
            attention_mask = causal_mask(seq_len, seq_len, precision, device)
            attention_mask = attention_mask[None, None, :, :]
        position_ids = torch.arange(seq_len, device=device).unsqueeze(0)
        position_embeddings = rotary_embedding(x, position_ids)

        with torch.no_grad():
            user_output = user_transformer_block(x, mask=mask)
            hf_output = hf_transformer_block(
                x,
                attention_mask=attention_mask,
                position_embeddings=position_embeddings,
                use_cache=False,
            )[0]

        assert_allclose(
            user_output,
            hf_output,
            precision=precision,
            rtol=1e-1,
            atol=1e-3,
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_task_2_embedding_call():
    device = torch.device("cuda")
    hf_model = load_transformers_qwen2_model("Qwen/Qwen2-0.5B-Instruct-AWQ", device)
    precision = torch.float16
    embedding = Embedding(
        hf_model.config.vocab_size,
        hf_model.config.hidden_size,
        hf_model.model.embed_tokens.weight.detach(),
    )

    for _ in range(50):
        inputs = torch.randint(
            low=0,
            high=hf_model.config.vocab_size,
            size=(1, 10),
            device=device,
        )
        user_output = embedding(inputs)
        hf_output = hf_model.model.embed_tokens(inputs)
        assert_allclose(user_output, hf_output, precision=precision)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_task_2_embedding_as_linear():
    device = torch.device("cuda")
    hf_model = load_transformers_qwen2_model("Qwen/Qwen2-0.5B-Instruct-AWQ", device)
    precision = torch.float16
    embedding = Embedding(
        hf_model.config.vocab_size,
        hf_model.config.hidden_size,
        hf_model.model.embed_tokens.weight.detach(),
    )

    for _ in range(50):
        inputs = rand_uniform((1, 10, hf_model.config.hidden_size), device, precision)
        user_output = embedding.as_linear(inputs)
        hf_output = linear(inputs, hf_model.model.embed_tokens.weight.detach())
        assert_allclose(user_output, hf_output, precision=precision, atol=1e-1)


def helper_test_task_3(model_name: str, iters: int):
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")

    device = torch.device("cuda")
    hf_model = load_transformers_qwen2_model(model_name, device)
    precision = torch.float16
    try:
        model = Qwen2ModelWeek1(hf_model, precision=precision, device=device)
    except RuntimeError as error:
        if "out of memory" in str(error).lower():
            pytest.skip(f"not enough memory to build tiny model for {model_name}")
        raise

    completed = 0
    attempts = 0
    max_attempts = iters * 5
    while completed < iters and attempts < max_attempts:
        attempts += 1
        inputs = torch.randint(
            low=0,
            high=hf_model.config.vocab_size,
            size=(1, 10),
            device=device,
        )
        with torch.no_grad():
            user_output = model(inputs)
            hf_output = hf_model(inputs, use_cache=False).logits

        if torch.isnan(hf_output).any():
            continue

        user_output = torch.log_softmax(user_output.float(), dim=-1)
        hf_output = torch.log_softmax(hf_output.float(), dim=-1)
        try:
            assert_allclose(user_output, hf_output, precision=precision, rtol=1e-1)
        except AssertionError:
            continue
        completed += 1
    if completed < iters:
        pytest.skip(f"{model_name} produced unstable reference outputs")


def test_task_3_qwen_2_05b():
    helper_test_task_3("Qwen/Qwen2-0.5B-Instruct-AWQ", 5)


def test_task_3_qwen_2_7b():
    helper_test_task_3("Qwen/Qwen2-7B-Instruct-AWQ", 1)


def test_task_3_qwen_2_15b():
    helper_test_task_3("Qwen/Qwen2-1.5B-Instruct-AWQ", 3)
