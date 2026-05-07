from abc import ABC, abstractmethod
from typing import Optional

import torch

from .attention import causal_mask


class TinyKvCache(ABC):
    @abstractmethod
    def update_and_fetch(
        self,
        key: torch.Tensor,
        value: torch.Tensor,
        mask_length: int | None = None,
        mask: torch.Tensor | str | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, int, Optional[torch.Tensor]]:
        """
        Update the key-value cache and fetch the full cached tensors.

        Shapes:
        - `key`: [B, H, L_new, D]
        - `value`: [B, H, L_new, D]
        - returns:
          - `key`: [B, H, L_total, D]
          - `value`: [B, H, L_total, D]
          - `offset`: `L_total`
        """


class BatchingKvCache(TinyKvCache):
    def __init__(self, max_active_requests: int, max_seq_len: int):
        self.max_active_requests = max_active_requests
        self.max_seq_len = max_seq_len
        self.kv_caches: list[TinyKvCache | None] = [None] * max_active_requests
        self.hd: tuple[int, int] | None = None

    def update_and_fetch(
        self,
        keys: torch.Tensor,
        values: torch.Tensor,
        mask_length: int | None = None,
        mask: torch.Tensor | str | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, int, Optional[torch.Tensor]]:
        batch_size, num_heads, seq_len, head_dim = keys.shape
        assert keys.shape == values.shape
        assert seq_len <= self.max_seq_len
        if self.hd is None:
            self.hd = (num_heads, head_dim)
        else:
            assert self.hd == (num_heads, head_dim), (
                f"expect {self.hd} but got {(num_heads, head_dim)}"
            )
        assert batch_size == self.max_active_requests
        assert mask_length is not None, "mask_length must be provided in batching mode"

        data: list[tuple[torch.Tensor, torch.Tensor, int, Optional[torch.Tensor]] | None] = []
        for batch_idx in range(batch_size):
            request_cache = self.kv_caches[batch_idx]
            if request_cache is None:
                data.append(None)
                continue
            key = keys[batch_idx : batch_idx + 1]
            value = values[batch_idx : batch_idx + 1]
            new_key, new_value, request_seq_len, request_mask = (
                request_cache.update_and_fetch(key, value)
            )
            data.append((new_key[0], new_value[0], request_seq_len, request_mask))

        def get_seq_len(
            item: tuple[torch.Tensor, torch.Tensor, int, Optional[torch.Tensor]] | None,
        ) -> int:
            if item is None:
                return 0
            return item[2]

        total_seq_len = max(map(get_seq_len, data))

        batched_keys = torch.zeros(
            (self.max_active_requests, num_heads, total_seq_len, head_dim),
            dtype=keys.dtype,
            device=keys.device,
        )
        batched_values = torch.zeros(
            (self.max_active_requests, num_heads, total_seq_len, head_dim),
            dtype=values.dtype,
            device=values.device,
        )
        masks = torch.full(
            (self.max_active_requests, mask_length, total_seq_len),
            -torch.inf,
            dtype=keys.dtype,
            device=keys.device,
        )

        for batch_idx in range(batch_size):
            item = data[batch_idx]
            if item is None:
                continue
            key, value, request_seq_len, request_mask = item
            batched_keys[batch_idx, :, total_seq_len - request_seq_len : total_seq_len, :] = key
            batched_values[batch_idx, :, total_seq_len - request_seq_len : total_seq_len, :] = value
            if request_mask is None or request_mask == "causal":
                masks[
                    batch_idx, :, total_seq_len - request_seq_len : total_seq_len
                ] = causal_mask(mask_length, request_seq_len, dtype=keys.dtype, device=keys.device)
            elif isinstance(request_mask, torch.Tensor):
                masks[
                    batch_idx, :, total_seq_len - request_seq_len : total_seq_len
                ] = request_mask
            else:
                raise NotImplementedError

        return (
            batched_keys,
            batched_values,
            None,
            masks.reshape(batch_size, 1, mask_length, total_seq_len),
        )

    def add_request(self, prefilled: TinyKvCache, id: int):
        if id >= self.max_active_requests:
            raise ValueError(f"Request id {id} is out of range")
        if getattr(prefilled, "key_values", None) is not None:
            keys, _ = prefilled.key_values
            batch_size, num_heads, _, head_dim = keys.shape
            assert batch_size == 1
            if self.hd is None:
                self.hd = (num_heads, head_dim)
            else:
                assert self.hd == (num_heads, head_dim)
        self.kv_caches[id] = prefilled

    def remove_request(self, id: int):
        if id >= self.max_active_requests:
            raise ValueError(f"Request id {id} is out of range")
        if self.kv_caches[id] is None:
            raise ValueError(f"Request id {id} is not in the cache")
        self.kv_caches[id] = None


class TinyKvFullCache(TinyKvCache):
    def __init__(self):
        self.key_values = None
        self.offset = 0

    def update_and_fetch(
        self,
        key: torch.Tensor,
        value: torch.Tensor,
        mask_length: int | None = None,
        mask: torch.Tensor | str | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, int, Optional[torch.Tensor]]:
        """
        Shapes:
        - `key`: [B, H, L_new, D]
        - `value`: [B, H, L_new, D]
        - returns:
          - `key`: [B, H, L_total, D]
          - `value`: [B, H, L_total, D]
          - `offset`: `L_total`
        """
        if self.key_values is None:
            assert self.offset == 0
            assert key.shape == value.shape
            self.key_values = (key, value)
            _, _, seq_len, _ = key.shape
            self.offset = seq_len
            return key, value, self.offset, mask
        else:
            batch_size, num_heads, seq_len, head_dim = key.shape
            assert key.shape == value.shape
            prev_keys, prev_values = self.key_values
            assert prev_keys.shape == (
                batch_size,
                num_heads,
                self.offset,
                head_dim,
            )
            assert prev_values.shape == (
                batch_size,
                num_heads,
                self.offset,
                head_dim,
            )
            new_keys = torch.cat([prev_keys, key], dim=2)
            new_values = torch.cat([prev_values, value], dim=2)
            self.key_values = (new_keys, new_values)
            self.offset += seq_len
            return new_keys, new_values, self.offset, mask

    def rewind(self, n: int):
        self.offset -= n
        self.key_values = (
            self.key_values[0][:, :, : self.offset],
            self.key_values[1][:, :, : self.offset],
        )
