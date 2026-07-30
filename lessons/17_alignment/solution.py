from torch.nn import functional as F


def pairwise_preference_loss(chosen_reward, rejected_reward):
    return -F.logsigmoid(chosen_reward - rejected_reward).mean()


def dpo_loss(policy_chosen, policy_rejected, reference_chosen, reference_rejected, beta=0.1):
    return -F.logsigmoid(
        beta * ((policy_chosen - policy_rejected) - (reference_chosen - reference_rejected))
    ).mean()
