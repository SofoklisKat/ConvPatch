"""Patch-embedding modules.

This is the *independent variable* of the ConvPatch study. Every variant must
expose the same interface so the downstream Transformer encoder is untouched:

    - input:  (B, C, H, W)
    - output: (B, N, D) tokens
    - attributes ``num_patches`` (int), ``grid_size`` (tuple[int, int]),
      ``embed_dim`` (int)

The token count N is held *constant* across all variants by forcing the total
stem stride to equal ``patch_size``. This isolates the effect of stem structure
from token-count / sequence-length effects (see DESIGN.md, threats to validity).

Variants (DESIGN.md design space):
    A. LinearPatchEmbed        - single stride-P, PxP conv (standard ViT)
    B. ConvStemPatchEmbed      - stacked stride-2 3x3 convs, total stride P
    C. OverlappingPatchEmbed   - single conv with kernel > stride (overlap)
    D. HierarchicalPatchEmbed  - per-stage downsample + refinement conv
    E. DWSepConvStemPatchEmbed - depthwise-separable conv stem (param-efficient)
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn


def _pair(x: int | tuple[int, int]) -> tuple[int, int]:
    return (x, x) if isinstance(x, int) else x


def _num_stride2_stages(patch_size: int) -> int:
    """Number of stride-2 stages needed to reach a total stride of ``patch_size``."""
    if patch_size < 1 or (patch_size & (patch_size - 1)) != 0:
        raise ValueError(
            f"patch_size must be a power of 2 for stacked stride-2 stems, got {patch_size}"
        )
    return int(math.log2(patch_size))


def _stage_channels(embed_dim: int, n_stages: int, in_chans: int, min_ch: int = 24) -> list[int]:
    """Channel schedule ramping from input up to ``embed_dim`` over ``n_stages``.

    Last stage outputs exactly ``embed_dim``; earlier stages are halved
    successively (floored at ``min_ch``). Returns a list of length n_stages + 1
    where the first element is ``in_chans``.
    """
    chs = [in_chans]
    for i in range(n_stages):
        c = embed_dim // (2 ** (n_stages - 1 - i))
        chs.append(max(c, min_ch) if i < n_stages - 1 else embed_dim)
    return chs


class _PatchEmbedBase(nn.Module):
    """Shared bookkeeping: validates sizes and exposes the standard interface."""

    def __init__(
        self,
        img_size: int | tuple[int, int],
        patch_size: int | tuple[int, int],
        embed_dim: int,
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

    def _check_input(self, x: torch.Tensor) -> None:
        _, _, h, w = x.shape
        if (h, w) != self.img_size:
            raise ValueError(f"input size {(h, w)} != expected img_size {self.img_size}")

    def _to_tokens(self, x: torch.Tensor) -> torch.Tensor:
        """(B, D, gh, gw) -> (B, N, D), asserting the grid matches num_patches."""
        b, d, gh, gw = x.shape
        if (gh, gw) != self.grid_size:
            raise RuntimeError(
                f"stem produced grid {(gh, gw)} but expected {self.grid_size}; "
                "total stride must equal patch_size"
            )
        return x.flatten(2).transpose(1, 2)


class LinearPatchEmbed(_PatchEmbedBase):
    """A. Standard ViT patchify stem: a single stride-P, PxP convolution.

    Equivalent to splitting the image into non-overlapping P x P patches,
    flattening each, and applying one linear projection to ``embed_dim``.
    """

    def __init__(
        self,
        img_size: int | tuple[int, int] = 224,
        patch_size: int | tuple[int, int] = 16,
        in_chans: int = 3,
        embed_dim: int = 768,
        norm_layer: type[nn.Module] | None = None,
    ) -> None:
        super().__init__(img_size, patch_size, embed_dim)
        self.proj = nn.Conv2d(
            in_chans, embed_dim, kernel_size=self.patch_size, stride=self.patch_size
        )
        self.norm = norm_layer(embed_dim) if norm_layer is not None else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self._check_input(x)
        x = self.proj(x)
        x = self._to_tokens(x)
        return self.norm(x)


def _conv_bn_act(in_ch: int, out_ch: int, k: int, s: int, act: bool = True) -> nn.Sequential:
    layers: list[nn.Module] = [
        nn.Conv2d(in_ch, out_ch, kernel_size=k, stride=s, padding=k // 2, bias=False),
        nn.BatchNorm2d(out_ch),
    ]
    if act:
        layers.append(nn.GELU())
    return nn.Sequential(*layers)


class ConvStemPatchEmbed(_PatchEmbedBase):
    """B. Lightweight convolutional stem: stacked stride-2 3x3 convs.

    log2(P) stride-2 blocks bring the spatial resolution down by exactly P, so
    the token count matches the linear baseline. A final 1x1 conv projects to
    ``embed_dim`` (a la Xiao et al., "Early Convolutions Help Transformers").
    """

    def __init__(
        self,
        img_size: int | tuple[int, int] = 224,
        patch_size: int | tuple[int, int] = 16,
        in_chans: int = 3,
        embed_dim: int = 768,
        norm_layer: type[nn.Module] | None = None,
    ) -> None:
        super().__init__(img_size, patch_size, embed_dim)
        n_stages = _num_stride2_stages(self.patch_size[0])
        if self.patch_size[0] != self.patch_size[1]:
            raise ValueError("ConvStemPatchEmbed assumes square patch_size")
        chs = _stage_channels(embed_dim, n_stages, in_chans)
        blocks = [_conv_bn_act(chs[i], chs[i + 1], k=3, s=2) for i in range(n_stages)]
        blocks.append(nn.Conv2d(chs[-1], embed_dim, kernel_size=1))
        self.proj = nn.Sequential(*blocks)
        self.norm = norm_layer(embed_dim) if norm_layer is not None else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self._check_input(x)
        x = self.proj(x)
        x = self._to_tokens(x)
        return self.norm(x)


class OverlappingPatchEmbed(_PatchEmbedBase):
    """C. Single conv with kernel > stride, so receptive fields overlap.

    Uses kernel = 2P, stride = P, padding = P//2 to keep the output grid at
    H/P x W/P (CvT/PVTv2-style overlapping patch embedding).
    """

    def __init__(
        self,
        img_size: int | tuple[int, int] = 224,
        patch_size: int | tuple[int, int] = 16,
        in_chans: int = 3,
        embed_dim: int = 768,
        norm_layer: type[nn.Module] | None = None,
    ) -> None:
        super().__init__(img_size, patch_size, embed_dim)
        p = self.patch_size[0]
        if self.patch_size[0] != self.patch_size[1]:
            raise ValueError("OverlappingPatchEmbed assumes square patch_size")
        self.proj = nn.Conv2d(
            in_chans, embed_dim, kernel_size=2 * p, stride=p, padding=p // 2
        )
        self.norm = norm_layer(embed_dim) if norm_layer is not None else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self._check_input(x)
        x = self.proj(x)
        x = self._to_tokens(x)
        return self.norm(x)


class HierarchicalPatchEmbed(_PatchEmbedBase):
    """D. Hierarchical stem: each stage downsamples then refines.

    Per stage: a stride-2 3x3 conv (downsample) followed by a stride-1 3x3 conv
    (refinement). More per-stage processing than the plain conv stem (B).
    """

    def __init__(
        self,
        img_size: int | tuple[int, int] = 224,
        patch_size: int | tuple[int, int] = 16,
        in_chans: int = 3,
        embed_dim: int = 768,
        norm_layer: type[nn.Module] | None = None,
    ) -> None:
        super().__init__(img_size, patch_size, embed_dim)
        n_stages = _num_stride2_stages(self.patch_size[0])
        if self.patch_size[0] != self.patch_size[1]:
            raise ValueError("HierarchicalPatchEmbed assumes square patch_size")
        chs = _stage_channels(embed_dim, n_stages, in_chans)
        stages: list[nn.Module] = []
        for i in range(n_stages):
            stages.append(
                nn.Sequential(
                    _conv_bn_act(chs[i], chs[i + 1], k=3, s=2),
                    _conv_bn_act(chs[i + 1], chs[i + 1], k=3, s=1),
                )
            )
        stages.append(nn.Conv2d(chs[-1], embed_dim, kernel_size=1))
        self.proj = nn.Sequential(*stages)
        self.norm = norm_layer(embed_dim) if norm_layer is not None else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self._check_input(x)
        x = self.proj(x)
        x = self._to_tokens(x)
        return self.norm(x)


def _dwsep_block(in_ch: int, out_ch: int) -> nn.Sequential:
    """Depthwise 3x3 stride-2 + pointwise 1x1, each followed by BN; GELU at end."""
    return nn.Sequential(
        nn.Conv2d(in_ch, in_ch, kernel_size=3, stride=2, padding=1, groups=in_ch, bias=False),
        nn.BatchNorm2d(in_ch),
        nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.GELU(),
    )


class DWSepConvStemPatchEmbed(_PatchEmbedBase):
    """E. Depthwise-separable conv stem: param-efficient counterpart to B."""

    def __init__(
        self,
        img_size: int | tuple[int, int] = 224,
        patch_size: int | tuple[int, int] = 16,
        in_chans: int = 3,
        embed_dim: int = 768,
        norm_layer: type[nn.Module] | None = None,
    ) -> None:
        super().__init__(img_size, patch_size, embed_dim)
        n_stages = _num_stride2_stages(self.patch_size[0])
        if self.patch_size[0] != self.patch_size[1]:
            raise ValueError("DWSepConvStemPatchEmbed assumes square patch_size")
        chs = _stage_channels(embed_dim, n_stages, in_chans)
        blocks = [_dwsep_block(chs[i], chs[i + 1]) for i in range(n_stages)]
        blocks.append(nn.Conv2d(chs[-1], embed_dim, kernel_size=1))
        self.proj = nn.Sequential(*blocks)
        self.norm = norm_layer(embed_dim) if norm_layer is not None else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self._check_input(x)
        x = self.proj(x)
        x = self._to_tokens(x)
        return self.norm(x)
