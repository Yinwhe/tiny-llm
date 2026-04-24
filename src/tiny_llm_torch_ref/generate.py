from typing import Any, Callable

import torch

from .kv_cache import TinyKvFullCache
from .qwen2_week1 import Qwen2ModelWeek1
from .qwen2_week2 import Qwen2ModelWeek2


def simple_generate(
    model: Qwen2ModelWeek1,
    tokenizer: Any,
    prompt: str,
    sampler: Callable[[torch.Tensor], torch.Tensor] | None,
) -> str:
    """
    Shapes:
    - prompt tokens: [L]
    - returns: generated text
    """

    def _step(model: Qwen2ModelWeek1, y: torch.Tensor) -> torch.Tensor:
        """
        Shapes:
        - `y`: [L]
        - returns: [1]
        """
        logits = model(y[None, :])
        logits = logits[:, -1, :]
        logprobs = logits - torch.logsumexp(logits, dim=-1, keepdim=True)
        if sampler is None:
            return torch.argmax(logprobs, dim=-1)
        return sampler(logprobs)

    # prefill with the prompt
    tokens = torch.tensor(
        tokenizer.encode(prompt, add_special_tokens=False),
        device=model.embedding.weight.device,
        dtype=torch.long,
    )
    detokenizer = tokenizer.detokenizer
    detokenizer.reset()

    # generate/decode
    while True:
        token = _step(model, tokens)
        tokens = torch.cat([tokens, token])
        if token.item() == tokenizer.eos_token_id:
            break

        detokenizer.add_token(token.item())
        print(detokenizer.last_segment, end="", flush=True)

    detokenizer.finalize()
    print(detokenizer.last_segment, end="", flush=True)
    return detokenizer.text


def simple_generate_with_kv_cache(
    model: Qwen2ModelWeek2,
    tokenizer: Any,
    prompt: str,
) -> str:
    """
    Shapes:
    - prompt tokens: [L]
    - returns: generated text
    """
    kv_cache = [TinyKvFullCache() for _ in range(model.num_hidden_layers)]

    def _step(
        model: Qwen2ModelWeek2,
        y: torch.Tensor,
        offset: int,
        kv_cache: list[TinyKvFullCache],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Shapes:
        - `y`: [L_new]
        - returns:
          - next token: [1]
          - logprobs: [V]
        """
        logits = model(y[None, :], offset, kv_cache)
        logits = logits[:, -1, :]
        logprobs = logits - torch.logsumexp(logits, dim=-1, keepdim=True)
        y = torch.argmax(logprobs, dim=-1)
        return y, logprobs.squeeze(0)

    # prefill with the prompt
    tokens = torch.tensor(
        tokenizer.encode(prompt, add_special_tokens=False),
        device=model.embedding.weight.device,
        dtype=torch.long,
    )
    detokenizer = tokenizer.detokenizer
    detokenizer.reset()
    offset = 0
    # generate/decode
    while True:
        token, _ = _step(model, tokens, offset, kv_cache)
        if token.item() == tokenizer.eos_token_id:
            break
        detokenizer.add_token(token.item())
        print(detokenizer.last_segment, end="", flush=True)
        # The first iteration of this loop is prefill. We want to add the offset to the prefilled token size.
        # Otherwise, we add the decoded token size (which is always 1).
        offset += tokens.numel()
        tokens = token

    detokenizer.finalize()
    print(detokenizer.last_segment, end="", flush=True)
    return detokenizer.text
