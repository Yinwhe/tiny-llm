import mlx.core as mx
from mlx_lm.tokenizer_utils import TokenizerWrapper
from .qwen2_week1 import Qwen2ModelWeek1
from .qwen2_week2 import Qwen2ModelWeek2
from typing import Callable


def simple_generate(
    model: Qwen2ModelWeek1,
    tokenizer: TokenizerWrapper,
    prompt: str,
    sampler: Callable[[mx.array], mx.array] | None,
) -> str:
    def _step(model: Qwen2ModelWeek1, y: mx.array, offset: int) -> mx.array:
        logits = model(y[None], offset)
        logits = logits[:, -1, :]
        if sampler is None:
            y = mx.argmax(logits, axis=-1)
        else:
            y = sampler(logits)
        return y

    # prefill with the prompt
    tokens = mx.array(tokenizer.encode(prompt, add_special_tokens=False))
    detokenizer = tokenizer.detokenizer
    detokenizer.reset()

    # generate/decode
    while True:
        token = _step(model, tokens, tokens.size)
        mx.eval(token)
        tokens = mx.concat([tokens, token])
        if token.item() == tokenizer.eos_token_id:
            break
        detokenizer.add_token(token.item())
        print(detokenizer.last_segment, end="", flush=False)
    


def simple_generate_with_kv_cache(
    model: Qwen2ModelWeek2, tokenizer: TokenizerWrapper, prompt: str
) -> str:
    pass


def batch_generate(
    model: any,
    tokenizer: TokenizerWrapper,
    prompts: list[str],
    max_seq_len=512,
    batch_size=5,
    prefill_step=128,
):
    pass
