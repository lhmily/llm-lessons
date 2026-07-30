import torch


def parameter_memory(
    parameters: int, bytes_per_parameter: int = 4, optimizer_multiplier: float = 4.0
) -> int:
    raise NotImplementedError("TODO")


def training_flops(parameters: int, tokens: int) -> int:
    raise NotImplementedError("TODO")


def fit_power_law(compute: torch.Tensor, loss: torch.Tensor, floor: float) -> tuple[float, float]:
    raise NotImplementedError("TODO")
