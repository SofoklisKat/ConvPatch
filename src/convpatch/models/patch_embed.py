"""Patch-embedding modules.

This is the *independent variable* of the ConvPatch study. Every variant must
expose the same interface so the downstream Transformer encoder is untouched:

    - input:  (B, C, H, W)
    - output: (B, N, D) tokens
    - attribute ``num_patches`` (int) and ``grid_size`` (tuple[int, int])

This file currently provides only the linear baseline (standard ViT). The
convolutional variants (B-E in DESIGN.md) will be added here later.
"""

from __future__ import annotations

import torch
import torch.nn as nn


def _pair(x: int | tuple[int, int]) -> tuple[int, int]:
    return (x, x) if isinstance(x, int) else x


class LinearPatchEmbed(nn.Module):
    """Standard ViT patchify stem: a single stride-P, PxP convolution.

    This is mathematically equivalent to splitting the image into
    non-overlapping P x P patches, flattening each, and applying one linear
    projection to dimension ``embed_dim``.
    """

    def __init__(
        self,
        img_size: int | tuple[int, int] = 224,
        patch_size: int | tuple[int, int] = 16,
        in_chans: int = 3,
        embed_dim: int = 768,
        norm_layer: type[nn.Module] | None = None,
    ) -> None:
        super().__init__()
        self.img_size = _pair(img_size)
        self.patch_size = _pair(patch_size)
        if self.img_size[0] % self.patch_size[0] or self.img_size[1] % self.patch_size[1]:
            raise ValueError(
                f"img_size {self.img_size} must be divisible by patch_size {self.patch_size}"
            )
        self.grid_size = (
            self.img_size[0] // self.patch_size[0],
            self.img_size[1] // self.patch_size[1],
        )
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.embed_dim = embed_dim

        self.proj = nn.Conv2d(
            in_chans, embed_dim, kernel_size=self.patch_size, stride=self.patch_size
        )
        self.norm = norm_layer(embed_dim) if norm_layer is not None else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, _, h, w = x.shape
        if (h, w) != self.img_size:
            raise ValueError(f"input size {(h, w)} != expected img_size {self.img_size}")
        x = self.proj(x)  # (B, D, H/P, W/P)
        x = x.flatten(2).transpose(1, 2)  # (B, N, D)
        x = self.norm(x)
        return x
