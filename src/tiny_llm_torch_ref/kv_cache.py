from abc import ABC, abstractmethod
from typing import Optional

import torch


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

    def update_and_fetch(
        self,
        keys: torch.Tensor,
        values: torch.Tensor,
        mask_length: int | None = None,
        mask: torch.Tensor | str | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, int, Optional[torch.Tensor]]:
        pass

    def add_request(self, prefilled: TinyKvCache, id: int):
        pass

    def remove_request(self, id: int):
        pass


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
