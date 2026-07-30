import torch
from torch.nn import functional as F


def append_cache(cache, k, v):
    return (k, v) if cache is None else (torch.cat((cache[0], k), -2), torch.cat((cache[1], v), -2))


def offset_causal_mask(query_length, key_length, past_length, device=None):
    return (
        torch.arange(key_length, device=device)[None, :]
        <= torch.arange(query_length, device=device)[:, None] + past_length
    )


def cached_attention(q, k, v, cache=None):
    past = 0 if cache is None else cache[0].size(-2)
    k, v = append_cache(cache, k, v)
    scores = q @ k.transpose(-2, -1) / (q.size(-1) ** 0.5)
    mask = offset_causal_mask(q.size(-2), k.size(-2), past, q.device)
    return F.softmax(scores.masked_fill(~mask, float("-inf")), -1) @ v, (k, v)
