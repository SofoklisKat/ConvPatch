"""Dataset and dataloader construction for CIFAR-10 / CIFAR-100.

ViTs trained from scratch on small images need fairly strong augmentation, so
the train pipeline uses RandomCrop + flip + RandAugment + normalization. The
eval pipeline is just normalization.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

CIFAR_STATS = {
    "cifar10": ((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    "cifar100": ((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762)),
}
NUM_CLASSES = {"cifar10": 10, "cifar100": 100}
_DATASET_CLS = {"cifar10": datasets.CIFAR10, "cifar100": datasets.CIFAR100}


@dataclass
class DataConfig:
    dataset: str = "cifar10"
    data_dir: str = "./data"
    img_size: int = 32
    batch_size: int = 128
    num_workers: int = 4
    randaugment: bool = True
    randaug_n: int = 2
    randaug_m: int = 9


def build_transforms(cfg: DataConfig):
    mean, std = CIFAR_STATS[cfg.dataset]
    train_tfms = [
        transforms.RandomCrop(cfg.img_size, padding=4, padding_mode="reflect"),
        transforms.RandomHorizontalFlip(),
    ]
    if cfg.randaugment:
        train_tfms.append(transforms.RandAugment(num_ops=cfg.randaug_n, magnitude=cfg.randaug_m))
    train_tfms += [transforms.ToTensor(), transforms.Normalize(mean, std)]

    eval_tfms = [transforms.ToTensor(), transforms.Normalize(mean, std)]
    if cfg.img_size != 32:
        eval_tfms = [transforms.Resize(cfg.img_size)] + eval_tfms
    return transforms.Compose(train_tfms), transforms.Compose(eval_tfms)


def build_loaders(cfg: DataConfig) -> tuple[DataLoader, DataLoader, int]:
    if cfg.dataset not in _DATASET_CLS:
        raise ValueError(f"unknown dataset {cfg.dataset!r}; choose from {list(_DATASET_CLS)}")
    ds_cls = _DATASET_CLS[cfg.dataset]
    train_tf, eval_tf = build_transforms(cfg)

    train_set = ds_cls(cfg.data_dir, train=True, download=True, transform=train_tf)
    test_set = ds_cls(cfg.data_dir, train=False, download=True, transform=eval_tf)

    pin = torch.cuda.is_available()
    train_loader = DataLoader(
        train_set,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=pin,
        drop_last=True,
        persistent_workers=cfg.num_workers > 0,
    )
    test_loader = DataLoader(
        test_set,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=pin,
        persistent_workers=cfg.num_workers > 0,
    )
    return train_loader, test_loader, NUM_CLASSES[cfg.dataset]
