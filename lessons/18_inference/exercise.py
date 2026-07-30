import torch


def quantize_int8(weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    raise NotImplementedError("TODO")


def dequantize_int8(values: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    raise NotImplementedError("TODO")


def storage_bytes(values: torch.Tensor, scale: torch.Tensor) -> int:
    raise NotImplementedError("TODO")


def dynamic_batches(lengths: list[int], max_requests: int, max_tokens: int) -> list[list[int]]:
    raise NotImplementedError("TODO")
