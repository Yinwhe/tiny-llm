from datetime import datetime
from typing import Any

import torch

from .kv_cache import BatchingKvCache, TinyKvFullCache


def _step(model: Any, y: torch.Tensor, offsets: list[int], kv_cache: Any):
    logits = model(y, offsets, kv_cache)
    logits = logits[:, -1, :]
    logprobs = logits - torch.logsumexp(logits, dim=-1, keepdim=True)
    sampler = lambda x: torch.argmax(x, dim=-1)
    return sampler(logprobs)


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
            device=model.embedding.weight.device,
        )
        self.prefill_max_step = prefill_max_step
        self.is_done = False
        self.is_prefill_done = False
        self.eos_token_id = tokenizer.eos_token_id
        self.next_token = None
        self.offset = 0
        self.prompt_idx = prompt_idx

    def try_prefill(self):
        """
        Prefill this request up to `prefill_max_step` tokens.
        """
        if self.is_prefill_done:
            raise ValueError("prefill called after done")
        tokens_to_prefill = min(
            self.prefill_max_step,
            self.prefill_tokens.numel() - self.offset,
        )
        token = _step(
            self.model,
            self.prefill_tokens[self.offset : self.offset + tokens_to_prefill][None],
            [self.offset],
            self.kv_cache,
        )
        self.offset += tokens_to_prefill
        if self.offset == self.prefill_tokens.numel():
            self.is_prefill_done = True
            self.decode_done(token.item(), update_offset=False)

    def decode_done(self, token, update_offset=True):
        if self.is_done:
            raise ValueError("decode called after done")
        if token == self.eos_token_id:
            self.is_done = True
            return
        self.detokenizer.add_token(token)
        self.next_token = token
        if update_offset:
            self.offset += 1

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
        percentage = (
            pending_prefill_request.offset
            / pending_prefill_request.prefill_tokens.numel()
        ) * 100
        print(
            f"{animation_frame} Prefill [req {pending_prefill_request.prompt_idx}]: {percentage:.2f}% ({pending_prefill_request.prefill_tokens.numel() - pending_prefill_request.offset} remaining tokens)",
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
                prefill_kv_cache = pending_prefill_request.kv_cache
                found_slot = False
                for i in range(batch_size):
                    if decode_requests[i] is None:
                        for prefill_cache, batch_cache in zip(
                            prefill_kv_cache, kv_cache
                        ):
                            batch_cache.add_request(prefill_cache, i)
                        decode_requests[i] = pending_prefill_request
                        found_slot = True
                        made_progress = True
                        break
                if found_slot:
                    pending_prefill_request = None
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
            for req in decode_requests:
                if req is not None:
                    next_tokens.append(req.next_token)
                    offsets.append(req.offset)
                else:
                    next_tokens.append(0)
                    offsets.append(0)
            next_tokens = torch.tensor(
                next_tokens,
                device=model.embedding.weight.device,
                dtype=torch.long,
            )
            next_tokens = _step(model, next_tokens.reshape(-1, 1), offsets, kv_cache)
            for i in range(batch_size):
                req = decode_requests[i]
                if req is not None:
                    req.decode_done(next_tokens[i].item())
                    remove_reason = None
                    if req.is_done:
                        remove_reason = "EOS"
                    elif req.offset >= max_seq_len:
                        remove_reason = "max seq len"
                    if remove_reason is not None:
                        print(f"Removing request {i} due to {remove_reason}", flush=True)
                        for layer_cache in kv_cache:
                            layer_cache.remove_request(i)
                        result.append((req.prompt_idx, req.text()))
                        decode_requests[i] = None
            _print_progress(
                decode_requests,
                pending_prefill_request,
                len(prompts),
                progress_cnt,
                start_time,
            )
            progress_cnt += 1
    return result
