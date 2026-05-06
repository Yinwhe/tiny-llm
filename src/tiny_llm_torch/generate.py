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
        pass


def simple_generate_with_kv_cache(
    model: Any,
    tokenizer: Any,
    prompt: str,
) -> str:
    def _step(model: Any, y: torch.Tensor, offset: int, kv_cache: Any):
        pass


def speculative_generate(
    draft_model: Any,
    model: Any,
    draft_tokenizer: Any,
    tokenizer: Any,
    prompt: str,
) -> str:
    pass
