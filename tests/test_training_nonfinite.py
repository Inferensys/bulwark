"""optimizer_step: a gradient-accumulation window without finite gradients is skipped, with
fp16 loss scaling on (as on a T4) and off (bf16/fp32), instead of crashing or writing NaN."""

from __future__ import annotations

import torch
from torch import nn

from bulwark.train.common import optimizer_step


def _setup(enabled: bool) -> tuple[nn.Parameter, nn.Parameter, torch.optim.Optimizer, torch.amp.GradScaler]:
    weight, loss_weight = nn.Parameter(torch.ones(3)), nn.Parameter(torch.zeros(1))
    optimizer = torch.optim.SGD([{"params": [weight]}, {"params": [loss_weight]}], lr=0.1)
    return weight, loss_weight, optimizer, torch.amp.GradScaler("cpu", enabled=enabled)


def test_scaled_windows_without_finite_gradients_are_skipped() -> None:
    weight, _, optimizer, scaler = _setup(enabled=True)
    # Every loss of the first window was non-finite: no backward ran, the scaler was never used.
    assert optimizer_step(optimizer, scaler, [weight], has_grads=False) is False
    scaler.scale((weight * float("nan")).sum()).backward()  # an fp16 overflow
    assert optimizer_step(optimizer, scaler, [weight], has_grads=True) is False
    assert torch.equal(weight.detach(), torch.ones(3)) and scaler.get_scale() == 2.0**15
    assert optimizer_step(optimizer, scaler, [weight], has_grads=False) is False  # after the scaler was used
    scaler.scale((weight * 2).sum()).backward()
    assert optimizer_step(optimizer, scaler, [weight], has_grads=True) is True
    assert (weight.detach() < 1).all() and weight.grad is None


def test_unscaled_non_finite_gradients_never_reach_the_weights() -> None:
    weight, loss_weight, optimizer, scaler = _setup(enabled=False)
    (weight.sum() + loss_weight.sum() * float("inf")).backward()  # only the loss weight blows up
    assert optimizer_step(optimizer, scaler, [weight], has_grads=True) is False
    assert torch.equal(weight.detach(), torch.ones(3)) and torch.equal(loss_weight.detach(), torch.zeros(1))
    (weight.sum() + loss_weight.sum()).backward()
    assert optimizer_step(optimizer, scaler, [weight], has_grads=True) is True
    assert (weight.detach() < 1).all()
