import torch
from torch.nn import functional as F

from llm_lessons.data import classification_data
from llm_lessons.testing import load_lesson

m = load_lesson(__file__)


def test_mlp_and_loss():
    torch.manual_seed(0)
    model = m.MLP(2, 16, 2)
    x, y = classification_data()
    logits = model(x)
    assert logits.shape == (128, 2)
    assert torch.allclose(m.cross_entropy(logits, y), F.cross_entropy(logits, y))
    opt = torch.optim.Adam(model.parameters(), lr=0.05)
    first = m.optimization_step(model, x, y, opt)
    for _ in range(40):
        last = m.optimization_step(model, x, y, opt)
    assert last < first * 0.25
