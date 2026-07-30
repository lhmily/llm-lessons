import torch


def cosine_similarity(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    denominator = a.norm(dim=-1) * b.norm(dim=-1)
    return (a * b).sum(dim=-1) / denominator.clamp_min(1e-8)


def batched_linear(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    return x @ weight + bias


def standardize(x: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    mean = x.mean(dim=-1, keepdim=True)
    variance = x.var(dim=-1, keepdim=True, unbiased=False)
    return (x - mean) / torch.sqrt(variance + eps)
