import torch
from torch import nn


def train_step(model, x, y, optimizer, max_norm=1.0):
    model.train()
    optimizer.zero_grad()
    _, loss = model(x, y)
    loss.backward()
    nn.utils.clip_grad_norm_(model.parameters(), max_norm)
    optimizer.step()
    return loss.item()


def evaluate(model, batches):
    was = model.training
    model.eval()
    with torch.no_grad():
        result = sum(model(x, y)[1].item() for x, y in batches) / len(batches)
    model.train(was)
    return result
