import torch

from llm_lessons.data import byte_tokens, make_batches
from llm_lessons.testing import load_lesson
from llm_lessons.tiny_gpt import TinyGPT, TinyGPTConfig

m = load_lesson(__file__)


def test_training_reduces_loss():
    torch.manual_seed(0)
    model = TinyGPT(TinyGPTConfig(context_length=8, d_model=16, n_heads=2, n_layers=1))
    batch = make_batches(
        byte_tokens(), batch_size=8, context_length=8, generator=torch.Generator().manual_seed(0)
    )
    opt = torch.optim.AdamW(model.parameters(), lr=0.02)
    first = m.evaluate(model, [batch])
    for _ in range(20):
        m.train_step(model, *batch, opt)
    assert m.evaluate(model, [batch]) < first * 0.7
