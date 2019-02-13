import torch

from torch import Tensor


def beta_smooth_l1_loss(input: Tensor, target: Tensor, beta: float) -> Tensor:
    diff = torch.abs(input - target)
    loss = torch.where(diff < beta, 0.5 * diff ** 2 / beta, diff - 0.5 * beta)
    loss = loss.sum() / (input.numel() + 1e-8)
    return loss
