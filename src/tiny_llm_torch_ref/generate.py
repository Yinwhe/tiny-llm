from typing import Any, Callable

import torch

from .qwen2_week1 import Qwen2ModelWeek1


def simple_generate(
    model: Qwen2ModelWeek1,
    tokenizer: Any,
    prompt: str,
    sampler: Callable[[torch.Tensor], torch.Tensor] | None,
) -> str:
    def _step(model: Qwen2ModelWeek1, y: torch.Tensor) -> torch.Tensor:
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
