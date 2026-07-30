from __future__ import annotations

import tempfile
from pathlib import Path

import torch

from llm_lessons.data import byte_tokens, make_batches
from llm_lessons.tiny_gpt import TinyGPT, TinyGPTConfig
from llm_lessons.utils import load_checkpoint, save_checkpoint


def test_end_to_end_forward_backward_and_checkpoint():
    torch.manual_seed(0)
    config = TinyGPTConfig(context_length=8, d_model=16, n_heads=2, n_layers=1)
    model = TinyGPT(config)
    x, y = make_batches(
        byte_tokens(),
        batch_size=2,
        context_length=config.context_length,
        generator=torch.Generator().manual_seed(1),
    )
    logits, loss = model(x, y)
    assert logits.shape == (2, 8, 256)
    assert loss is not None and torch.isfinite(loss)
    loss.backward()
    assert all(parameter.grad is not None for parameter in model.parameters())

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    with tempfile.TemporaryDirectory() as folder:
        path = Path(folder) / "model.pt"
        save_checkpoint(path, model, optimizer, step=7)
        restored = TinyGPT(config)
        assert load_checkpoint(path, restored) == 7
        expected, _ = model(x)
        actual, _ = restored(x)
        assert torch.allclose(expected, actual)
