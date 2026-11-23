"""Muon for weight matrices (my implementation of the published recipe: momentum, then the update is
replaced by an approximately orthogonal matrix via five Newton-Schulz steps). Other parameters stay
on Adam. Matrices with a leading expert dimension are orthogonalised one expert at a time."""
from __future__ import annotations

import torch


def _orthogonalise(g, steps=5):
    a, b, c = 3.4445, -4.7750, 2.0315
    x = g / (g.flatten(-2).norm(dim=-1)[..., None, None] + 1e-7)
    tall = x.shape[-2] > x.shape[-1]
    if tall:
        x = x.transpose(-1, -2)
    for _ in range(steps):
        A = x @ x.transpose(-1, -2)
        x = a * x + (b * A + c * A @ A) @ x
    return x.transpose(-1, -2) if tall else x


class Muon(torch.optim.Optimizer):
    def __init__(self, params, lr=0.02, momentum=0.95, weight_decay=0.0):
        super().__init__(params, dict(lr=lr, momentum=momentum, weight_decay=weight_decay))

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                buf = self.state[p].setdefault("buf", torch.zeros_like(p))
                buf.mul_(group["momentum"]).add_(p.grad)
                u = _orthogonalise(p.grad.add(buf, alpha=group["momentum"]))     # Nesterov
                if p.ndim == 3:
                    # an expert that saw no tokens this step has no gradient; orthogonalising its stale
                    # momentum would still give it a full-size update, so leave it alone
                    u = u * (p.grad.flatten(1).abs().sum(1) > 0)[:, None, None]
                scale = max(1.0, p.shape[-2] / p.shape[-1]) ** 0.5
                if group["weight_decay"]:
                    # decay at the Adam-scale rate so Muon and AdamW arms shrink weights equally per step
                    p.mul_(1 - group.get("wd_lr", group["lr"]) * group["weight_decay"])
                p.add_(u, alpha=-group["lr"] * scale)
