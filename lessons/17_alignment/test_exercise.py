import torch

from llm_lessons.testing import load_lesson

m = load_lesson(__file__)


def test_losses():
    assert m.pairwise_preference_loss(
        torch.tensor([10.0]), torch.tensor([-10.0])
    ) < m.pairwise_preference_loss(torch.tensor([-10.0]), torch.tensor([10.0]))
    assert torch.allclose(
        m.dpo_loss(
            torch.tensor([1.0]), torch.tensor([0.0]), torch.tensor([1.0]), torch.tensor([0.0])
        ),
        torch.tensor(0.6931472),
    )
