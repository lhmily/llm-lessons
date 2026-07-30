import torch


def parameter_memory(parameters, bytes_per_parameter=4, optimizer_multiplier=4.0):
    return int(parameters * bytes_per_parameter * optimizer_multiplier)


def training_flops(parameters, tokens):
    return 6 * parameters * tokens


def fit_power_law(compute, loss, floor):
    x = torch.log(compute.double())
    y = torch.log((loss.double() - floor).clamp_min(1e-12))
    slope = ((x - x.mean()) * (y - y.mean())).sum() / ((x - x.mean()) ** 2).sum()
    return torch.exp(y.mean() - slope * x.mean()).item(), slope.item()
