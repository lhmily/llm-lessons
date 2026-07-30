"""General utilities used after the model-building lessons."""

from __future__ import annotations

from pathlib import Path

import torch
from torch import nn


def parameter_count(module: nn.Module, *, trainable_only: bool = False) -> int:
    parameters = module.parameters()
    if trainable_only:
        parameters = (parameter for parameter in parameters if parameter.requires_grad)
    return sum(parameter.numel() for parameter in parameters)


def save_checkpoint(path: str | Path, model: nn.Module, optimizer=None, step: int = 0) -> None:
    payload = {"model": model.state_dict(), "step": step}
    if optimizer is not None:
        payload["optimizer"] = optimizer.state_dict()
    torch.save(payload, path)


def load_checkpoint(path: str | Path, model: nn.Module, optimizer=None) -> int:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    model.load_state_dict(payload["model"])
    if optimizer is not None and "optimizer" in payload:
        optimizer.load_state_dict(payload["optimizer"])
    return int(payload["step"])
