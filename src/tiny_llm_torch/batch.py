from datetime import datetime
from typing import Any

import torch

from .kv_cache import BatchingKvCache, TinyKvFullCache


def _step(model: Any, y: torch.Tensor, offsets: list[int], kv_cache: Any):
    logits = model(y, offsets, kv_cache)
    logits = logits[:, -1, :]
    logprobs = logits - torch.logsumexp(logits, dim=-1, keepdim=True)
    sampler = lambda x: torch.argmax(x, dim=-1)
    y = sampler(logprobs)
    return y


class Request:
    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        prompt: str,
        prefill_max_step: int = 128,
        prompt_idx: int = 0,
    ):
        self.prompt = prompt
        self.kv_cache = [TinyKvFullCache() for _ in range(model.num_hidden_layers)]
        self.model = model
        self.detokenizer = tokenizer.detokenizer.__class__(tokenizer._tokenizer)
        self.prefill_tokens = torch.tensor(
            tokenizer.encode(prompt, add_special_tokens=False),
            dtype=torch.long,
        )
        self.prefill_max_step = prefill_max_step
        self.is_done = False
        self.is_prefill_done = False
        self.eos_token_id = tokenizer.eos_token_id
        self.next_token = None
        self.offset = 0
        self.prompt_idx = prompt_idx

    def try_prefill(self):
        pass

    def decode_done(self, token, update_offset=True):
        if self.is_done:
            raise ValueError("decode called after done")
        if token == self.eos_token_id:
            self.is_done = True
            return
        pass

    def text(self):
        return self.detokenizer.text


def _print_progress(
    requests: list[Request | None],
    pending_prefill_request: Request | None,
    queue_size: int,
    progress_cnt: int,
    start_time: datetime,
):
    print(f"  --- {datetime.now() - start_time}")
    animation_frames = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]
    animation_frame = animation_frames[progress_cnt % len(animation_frames)]
    for i, request in enumerate(requests):
        if request is None:
            print(f"  Decode #{i}: idle", flush=True)
        else:
            text_preview = request.text()[-80:].replace("\n", " ")
            print(
                f"{animation_frame} Decode [req {request.prompt_idx}, {request.offset}]: {text_preview}",
                flush=True,
            )
    if pending_prefill_request is not None:
        if pending_prefill_request.is_prefill_done:
            print(
                f"  Prefill [req {pending_prefill_request.prompt_idx}]: done, waiting for slot, {queue_size} requests in queue",
                flush=True,
            )
            return
        precentage = (
            pending_prefill_request.offset / pending_prefill_request.prefill_tokens.numel()
        ) * 100
        print(
            f"{animation_frame} Prefill [req {pending_prefill_request.prompt_idx}]: {precentage:.2f}% ({pending_prefill_request.prefill_tokens.numel() - pending_prefill_request.offset} remaining tokens)",
            flush=True,
        )
    else:
        print(f"  Prefill: idle, {queue_size} requests in queue", flush=True)


def batch_generate(
    model: Any,
    tokenizer: Any,
    prompts: list[str],
    max_seq_len=512,
    batch_size=5,
    prefill_step=128,
):
    decode_requests: list[Request | None] = [None] * batch_size
    kv_cache = [
        BatchingKvCache(max_active_requests=batch_size, max_seq_len=max_seq_len)
        for _ in range(model.num_hidden_layers)
    ]
    result = []
    pending_prefill_request = None
    next_request_idx = 0
    progress_cnt = 0
    start_time = datetime.now()

    while True:
        if len(prompts) == 0 and all(req is None for req in decode_requests):
            break
        if len(prompts) > 0 and pending_prefill_request is None:
            prompt = prompts.pop(0)
            pending_prefill_request = Request(
                model, tokenizer, prompt, prefill_step, next_request_idx
            )
            next_request_idx += 1

        if pending_prefill_request is not None:
            made_progress = False
            if not pending_prefill_request.is_prefill_done:
                pending_prefill_request.try_prefill()
                made_progress = True
            if pending_prefill_request.is_prefill_done:
                pass
            if made_progress:
                _print_progress(
                    decode_requests,
                    pending_prefill_request,
                    len(prompts),
                    progress_cnt,
                    start_time,
                )
                progress_cnt += 1

        if any(req is not None for req in decode_requests):
            next_tokens = []
            offsets = []
            next_tokens = _step(model, next_tokens.reshape(-1, 1), offsets, kv_cache)
            for i in range(batch_size):
                pass
            _print_progress(
                decode_requests,
                pending_prefill_request,
                len(prompts),
                progress_cnt,
                start_time,
            )
            progress_cnt += 1
    return result
