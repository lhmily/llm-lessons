import torch

from llm_lessons.testing import load_lesson
from llm_lessons.tiny_gpt import TinyGPTConfig

m = load_lesson(__file__)


def test_tiny_gpt():
    torch.manual_seed(0)
    config = TinyGPTConfig(vocab_size=23, context_length=8, d_model=16, n_heads=4, n_layers=2)
    model = m.TinyGPT(config)
    tokens = torch.randint(0, 23, (2, 7))
    logits, loss = model(tokens, tokens)
    assert logits.shape == (2, 7, 23)
    assert loss.ndim == 0 and torch.isfinite(loss)
    assert model.lm_head.weight.data_ptr() == model.token_embedding.weight.data_ptr()
    loss.backward()
    assert m.parameter_count(model) > 0
