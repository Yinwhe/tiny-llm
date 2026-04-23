import torch

from .basics import linear


class Embedding:
    def __init__(self, vocab_size: int, embedding_dim: int, weight: torch.Tensor):
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.weight = weight

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return self.weight[x, :]

    def as_linear(self, x: torch.Tensor) -> torch.Tensor:
        return linear(x, self.weight)
