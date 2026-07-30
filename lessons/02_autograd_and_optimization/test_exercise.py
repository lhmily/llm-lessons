import torch

from llm_lessons.testing import load_lesson

m = load_lesson(__file__)


def test_gradients_and_training():
    v = torch.tensor([1.0, -2.0], requires_grad=True)
    loss = (v**2).sum()
    loss.backward()
    numeric = m.numerical_gradient(lambda z: (z**2).sum(), v.detach())
    assert torch.allclose(numeric, v.grad, atol=4e-3)
    x = torch.arange(8, dtype=torch.float32).view(-1, 1)
    y = 3 * x + 2
    w = torch.zeros(1, 1, requires_grad=True)
    b = torch.zeros(1, requires_grad=True)
    first = m.train_step(x, y, w, b, 0.01)
    for _ in range(100):
        last = m.train_step(x, y, w, b, 0.01)
    assert last < first * 0.01
