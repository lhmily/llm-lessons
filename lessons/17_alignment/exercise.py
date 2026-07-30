import torch


def pairwise_preference_loss(
    chosen_reward: torch.Tensor, rejected_reward: torch.Tensor
) -> torch.Tensor:
    raise NotImplementedError("TODO")


def dpo_loss(
    policy_chosen: torch.Tensor,
    policy_rejected: torch.Tensor,
    reference_chosen: torch.Tensor,
    reference_rejected: torch.Tensor,
    beta: float = 0.1,
) -> torch.Tensor:
    raise NotImplementedError("TODO")
