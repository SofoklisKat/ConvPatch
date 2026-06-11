"""Single-epoch train / eval loops with optional AMP and mixup."""

from __future__ import annotations

import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .utils import AverageMeter, Mixup, WarmupCosineLR, accuracy


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: WarmupCosineLR,
    device: torch.device,
    scaler: torch.amp.GradScaler | None = None,
    mixup: Mixup | None = None,
    grad_clip: float | None = None,
    amp: bool = True,
    log_interval: int = 50,
) -> dict[str, float]:
    model.train()
    loss_meter = AverageMeter()
    data_t, batch_t = AverageMeter(), AverageMeter()
    end = time.time()
    lr = optimizer.param_groups[0]["lr"]

    for i, (images, targets) in enumerate(loader):
        data_t.update(time.time() - end)
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        if mixup is not None:
            images, targets = mixup(images, targets)

        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, enabled=amp):
            outputs = model(images)
            loss = criterion(outputs, targets)

        if scaler is not None:
            scaler.scale(loss).backward()
            if grad_clip is not None:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if grad_clip is not None:
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

        lr = scheduler.step()
        loss_meter.update(loss.item(), images.size(0))
        batch_t.update(time.time() - end)
        end = time.time()

        if log_interval and (i % log_interval == 0 or i == len(loader) - 1):
            print(
                f"  iter {i:4d}/{len(loader)} | loss {loss_meter.avg:.4f} | lr {lr:.2e} "
                f"| {batch_t.avg*1e3:.0f} ms/it (data {data_t.avg*1e3:.0f} ms)",
                flush=True,
            )

    return {"loss": loss_meter.avg, "lr": lr, "ms_per_iter": batch_t.avg * 1e3}


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    amp: bool = True,
) -> dict[str, float]:
    model.eval()
    top1, top5 = AverageMeter(), AverageMeter()
    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, enabled=amp):
            outputs = model(images)
        acc1, acc5 = accuracy(outputs, targets, topk=(1, 5))
        top1.update(acc1, images.size(0))
        top5.update(acc5, images.size(0))
    return {"top1": top1.avg, "top5": top5.avg}
