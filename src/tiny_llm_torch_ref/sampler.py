import torch


def make_sampler(temp: float, top_p: float | None, top_k: int | None):
    def sample(logprobs: torch.Tensor):
        if temp == 0:
            return torch.argmax(logprobs, dim=-1)

        logprobs = logprobs.clone()

        if top_k is not None and top_k > 0:
            sorted_idx = torch.argsort(-logprobs, dim=-1)
            mask_elements = sorted_idx[:, top_k:]
            logprobs.scatter_(-1, mask_elements, -torch.inf)

        if top_p is not None and top_p > 0:
            sorted_idx = torch.argsort(-logprobs, dim=-1)
            sorted_logprobs = torch.take_along_dim(logprobs, sorted_idx, dim=-1)
            cumsum = torch.cumsum(torch.exp(sorted_logprobs), dim=-1)
            keep_sorted = cumsum < top_p
            keep_sorted[..., 0] = True
            logprobs.scatter_(
                -1,
                sorted_idx,
                torch.where(keep_sorted, sorted_logprobs, -torch.inf),
            )

        return torch.distributions.Categorical(logits=logprobs / temp).sample()

    return sample
