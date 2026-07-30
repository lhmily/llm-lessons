import torch
from torch.nn import functional as F

from llm_lessons.testing import load_lesson

m = load_lesson(__file__)


def test_examples_and_loss():
    x, y = m.make_examples(torch.arange(7), 3)
    assert x.tolist() == [[0, 1, 2], [1, 2, 3], [2, 3, 4], [3, 4, 5]]
    assert torch.equal(y, x + 1)
    train, val = m.split_tokens(torch.arange(10), 0.8)
    assert len(train) == 8 and len(val) == 2
    logits = torch.randn(2, 3, 5)
    targets = torch.randint(0, 5, (2, 3))
    assert torch.allclose(
        m.language_model_loss(logits, targets),
        F.cross_entropy(logits.view(-1, 5), targets.view(-1)),
    )
