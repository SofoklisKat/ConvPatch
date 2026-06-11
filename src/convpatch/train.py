"""Config-driven training entry point for the ConvPatch study.

Usage:
    python -m convpatch.train --config configs/cifar10_vit_tiny_linear.yaml
    python -m convpatch.train --config <cfg> --set train.epochs=5 train.amp=false
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import time
from pathlib import Path

import torch
import torch.nn as nn
import yaml

from .build import build_model
from .data import DataConfig, build_loaders
from .engine import evaluate, train_one_epoch
from .utils import (
    Mixup,
    MixupConfig,
    SoftTargetCrossEntropy,
    WarmupCosineLR,
    set_seed,
)


def _set_nested(cfg: dict, dotted: str, value: str) -> None:
    keys = dotted.split(".")
    d = cfg
    for k in keys[:-1]:
        d = d.setdefault(k, {})
    try:
        parsed = yaml.safe_load(value)
    except yaml.YAMLError:
        parsed = value
    d[keys[-1]] = parsed


def load_config(path: str, overrides: list[str] | None) -> dict:
    with open(path) as f:
        cfg = yaml.safe_load(f)
    for ov in overrides or []:
        key, _, val = ov.partition("=")
        _set_nested(cfg, key.strip(), val.strip())
    return cfg


def resolve_device(requested: str) -> torch.device:
    if requested == "cpu":
        return torch.device("cpu")
    if requested.startswith("cuda") and not torch.cuda.is_available():
        print("[warn] CUDA requested but unavailable; falling back to CPU")
        return torch.device("cpu")
    return torch.device(requested)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--set", nargs="*", default=[], help="dotted overrides key=value")
    args = parser.parse_args()

    cfg = load_config(args.config, args.set)
    tcfg = cfg["train"]
    seed = cfg.get("seed", 0)
    set_seed(seed, deterministic=tcfg.get("deterministic", False))

    device = resolve_device(tcfg.get("device", "cuda:0"))
    amp = bool(tcfg.get("amp", True)) and device.type == "cuda"

    run_name = cfg.get("run_name") or Path(args.config).stem
    out_dir = Path(cfg.get("out_dir", "runs")) / f"{run_name}_seed{seed}"
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "config.json", "w") as f:
        json.dump(cfg, f, indent=2)

    dcfg = DataConfig(
        dataset=cfg["data"]["dataset"],
        data_dir=cfg["data"].get("data_dir", "./data"),
        img_size=cfg["data"]["img_size"],
        batch_size=cfg["data"]["batch_size"],
        num_workers=cfg["data"].get("num_workers", 4),
        randaugment=cfg["data"].get("randaugment", True),
        randaug_n=cfg["data"].get("randaug_n", 2),
        randaug_m=cfg["data"].get("randaug_m", 9),
    )
    train_loader, test_loader, num_classes = build_loaders(dcfg)

    model = build_model(cfg, num_classes).to(device)
    n_params = model.num_params()
    print(
        f"Run: {run_name} | device {device} | amp {amp} | params {n_params/1e6:.2f}M "
        f"| tokens {model.patch_embed.num_patches} | classes {num_classes}",
        flush=True,
    )

    use_mixup = tcfg.get("mixup", True)
    if use_mixup:
        mixup = Mixup(
            MixupConfig(
                mixup_alpha=tcfg.get("mixup_alpha", 0.8),
                cutmix_alpha=tcfg.get("cutmix_alpha", 1.0),
                prob=tcfg.get("mixup_prob", 1.0),
                switch_prob=tcfg.get("mixup_switch_prob", 0.5),
                num_classes=num_classes,
                label_smoothing=tcfg.get("label_smoothing", 0.1),
            )
        )
        criterion: nn.Module = SoftTargetCrossEntropy()
    else:
        mixup = None
        criterion = nn.CrossEntropyLoss(label_smoothing=tcfg.get("label_smoothing", 0.1))

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=tcfg["lr"],
        weight_decay=tcfg.get("weight_decay", 0.05),
        betas=tuple(tcfg.get("betas", (0.9, 0.999))),
    )
    epochs = tcfg["epochs"]
    iters_per_epoch = len(train_loader)
    scheduler = WarmupCosineLR(
        optimizer,
        base_lr=tcfg["lr"],
        warmup_iters=tcfg.get("warmup_epochs", 5) * iters_per_epoch,
        total_iters=epochs * iters_per_epoch,
        min_lr=tcfg.get("min_lr", 1e-6),
    )
    scaler = torch.amp.GradScaler("cuda", enabled=amp)

    log_path = out_dir / "log.csv"
    with open(log_path, "w", newline="") as f:
        csv.writer(f).writerow(["epoch", "train_loss", "lr", "top1", "top5", "ms_per_iter", "epoch_s"])

    best_top1 = 0.0
    for epoch in range(epochs):
        t0 = time.time()
        print(f"Epoch {epoch+1}/{epochs}", flush=True)
        tr = train_one_epoch(
            model, train_loader, criterion, optimizer, scheduler, device,
            scaler=scaler if amp else None,
            mixup=mixup,
            grad_clip=tcfg.get("grad_clip", 1.0),
            amp=amp,
            log_interval=tcfg.get("log_interval", 50),
        )
        ev = evaluate(model, test_loader, device, amp=amp)
        epoch_s = time.time() - t0
        print(
            f"  -> top1 {ev['top1']:.2f} | top5 {ev['top5']:.2f} | "
            f"train_loss {tr['loss']:.4f} | epoch {epoch_s:.1f}s",
            flush=True,
        )
        with open(log_path, "a", newline="") as f:
            csv.writer(f).writerow(
                [epoch + 1, f"{tr['loss']:.4f}", f"{tr['lr']:.6e}",
                 f"{ev['top1']:.4f}", f"{ev['top5']:.4f}",
                 f"{tr['ms_per_iter']:.1f}", f"{epoch_s:.1f}"]
            )

        ckpt = {
            "epoch": epoch + 1,
            "model": model.state_dict(),
            "top1": ev["top1"],
            "config": cfg,
        }
        torch.save(ckpt, out_dir / "last.pt")
        if ev["top1"] > best_top1:
            best_top1 = ev["top1"]
            torch.save(ckpt, out_dir / "best.pt")

    print(f"Done. Best top1 = {best_top1:.2f} | artifacts in {out_dir}", flush=True)
    with open(out_dir / "summary.json", "w") as f:
        json.dump({"best_top1": best_top1, "params_M": n_params / 1e6,
                   "tokens": model.patch_embed.num_patches, "epochs": epochs}, f, indent=2)


if __name__ == "__main__":
    main()
