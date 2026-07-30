import torch
from torch import nn

from llm_lessons.testing import load_lesson

m = load_lesson(__file__)


def test_embeddings():
    table = torch.randn(7, 6)
    tokens = torch.tensor([[1, 4, 2]])
    assert torch.allclose(m.one_hot_lookup(tokens, table), table[tokens])
    positions = m.sinusoidal_positions(5, 6)
    assert positions.shape == (5, 6)
    assert torch.allclose(positions[0, 0::2], torch.zeros(3))
    tok = nn.Embedding(7, 6)
    pos = nn.Embedding(8, 6)
    output = m.embed_sequence(tokens, tok, pos)
    assert output.shape == (1, 3, 6) and not torch.allclose(output[:, 0], output[:, 1])
