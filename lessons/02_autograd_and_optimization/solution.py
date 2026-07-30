import torch


def mse(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return ((prediction - target) ** 2).mean()


def train_step(
    x: torch.Tensor, y: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, lr: float
) -> torch.Tensor:
    if weight.grad is not None:
        weight.grad.zero_()
    if bias.grad is not None:
        bias.grad.zero_()
    loss = mse(x @ weight + bias, y)
    loss.backward()
    with torch.no_grad():
        weight -= lr * weight.grad
        bias -= lr * bias.grad
    return loss.detach()


def numerical_gradient(function, value: torch.Tensor, epsilon: float = 1e-4) -> torch.Tensor:
    result = torch.empty_like(value)
    for i in range(value.numel()):
        plus = value.clone()
        minus = value.clone()
        plus[i] += epsilon
        minus[i] -= epsilon
        result[i] = (function(plus) - function(minus)) / (2 * epsilon)
    return result
