"""Training utilities: reproducibility, metrics, LR schedule, mixup/cutmix."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass

import numpy as np
import torch


def set_seed(seed: int, deterministic: bool = False) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.benchmark = True


class AverageMeter:
    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.sum = 0.0
        self.count = 0

    def update(self, val: float, n: int = 1) -> None:
        self.sum += val * n
        self.count += n

    @property
    def avg(self) -> float:
        return self.sum / max(self.count, 1)


@torch.no_grad()
def accuracy(output: torch.Tensor, target: torch.Tensor, topk=(1,)) -> list[float]:
    """Top-k accuracy (%) for hard integer targets."""
    maxk = max(topk)
    batch = target.size(0)
    _, pred = output.topk(maxk, dim=1, largest=True, sorted=True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0)
        res.append((correct_k * 100.0 / batch).item())
    return res


class WarmupCosineLR:
    """Linear warmup then cosine decay to ``min_lr``, stepped per iteration."""

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        base_lr: float,
        warmup_iters: int,
        total_iters: int,
        min_lr: float = 1e-6,
    ) -> None:
        self.opt = optimizer
        self.base_lr = base_lr
        self.warmup_iters = max(warmup_iters, 1)
        self.total_iters = total_iters
        self.min_lr = min_lr
        self.it = 0
        self.step()  # set initial lr

    def step(self) -> float:
        if self.it < self.warmup_iters:
            lr = self.base_lr * (self.it + 1) / self.warmup_iters
        else:
            progress = (self.it - self.warmup_iters) / max(
                self.total_iters - self.warmup_iters, 1
            )
            progress = min(progress, 1.0)
            lr = self.min_lr + 0.5 * (self.base_lr - self.min_lr) * (
                1 + math.cos(math.pi * progress)
            )
        for group in self.opt.param_groups:
            group["lr"] = lr
        self.it += 1
        return lr


@dataclass
class MixupConfig:
    mixup_alpha: float = 0.8
    cutmix_alpha: float = 1.0
    prob: float = 1.0
    switch_prob: float = 0.5
    num_classes: int = 10
    label_smoothing: float = 0.1


def _one_hot(target: torch.Tensor, num_classes: int, smoothing: float) -> torch.Tensor:
    off = smoothing / num_classes
    on = 1.0 - smoothing + off
    y = torch.full((target.size(0), num_classes), off, device=target.device)
    return y.scatter_(1, target.unsqueeze(1), on)


def _rand_bbox(h: int, w: int, lam: float) -> tuple[int, int, int, int]:
    ratio = math.sqrt(1.0 - lam)
    cut_h, cut_w = int(h * ratio), int(w * ratio)
    cy, cx = random.randint(0, h - 1), random.randint(0, w - 1)
    y1, y2 = max(cy - cut_h // 2, 0), min(cy + cut_h // 2, h)
    x1, x2 = max(cx - cut_w // 2, 0), min(cx + cut_w // 2, w)
    return y1, y2, x1, x2


class Mixup:
    """Batch-level Mixup / CutMix producing soft (label-smoothed) targets."""

    def __init__(self, cfg: MixupConfig) -> None:
        self.cfg = cfg

    def __call__(self, x: torch.Tensor, target: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        c = self.cfg
        y = _one_hot(target, c.num_classes, c.label_smoothing)
        if random.random() > c.prob:
            return x, y
        perm = torch.randperm(x.size(0), device=x.device)
        use_cutmix = random.random() < c.switch_prob
        if use_cutmix and c.cutmix_alpha > 0:
            lam = np.random.beta(c.cutmix_alpha, c.cutmix_alpha)
            y1, y2, x1, x2 = _rand_bbox(x.size(2), x.size(3), lam)
            x[:, :, y1:y2, x1:x2] = x[perm, :, y1:y2, x1:x2]
            lam = 1.0 - ((y2 - y1) * (x2 - x1) / (x.size(2) * x.size(3)))
        elif c.mixup_alpha > 0:
            lam = np.random.beta(c.mixup_alpha, c.mixup_alpha)
            x = lam * x + (1.0 - lam) * x[perm]
        else:
            return x, y
        y = lam * y + (1.0 - lam) * y[perm]
        return x, y


class SoftTargetCrossEntropy(torch.nn.Module):
    def forward(self, x: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return torch.sum(-target * torch.log_softmax(x, dim=-1), dim=-1).mean()
