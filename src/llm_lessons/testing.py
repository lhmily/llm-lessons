"""Helpers shared by lesson tests."""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path
from types import ModuleType

import torch


def seed_everything(seed: int = 0) -> torch.Generator:
    """Seed PyTorch and return an independent CPU generator."""
    torch.manual_seed(seed)
    return torch.Generator().manual_seed(seed)


def load_lesson(test_file: str) -> ModuleType:
    """Load exercise.py, or solution.py when LESSON_IMPL=solution."""
    folder = Path(test_file).resolve().parent
    implementation = "solution" if os.getenv("LESSON_IMPL") == "solution" else "exercise"
    path = folder / f"{implementation}.py"
    spec = importlib.util.spec_from_file_location(f"{folder.name}_{implementation}", path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def finite_difference(function, value: torch.Tensor, epsilon: float = 1e-4) -> torch.Tensor:
    """Estimate the gradient of a scalar function at a 1-D tensor."""
    gradient = torch.empty_like(value)
    for index in range(value.numel()):
        plus, minus = value.clone(), value.clone()
        plus[index] += epsilon
        minus[index] -= epsilon
        gradient[index] = (function(plus) - function(minus)) / (2 * epsilon)
    return gradient
