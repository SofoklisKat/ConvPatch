"""Build a ViT from a plain config dict.

The ``patch_embed.type`` field selects the patch-embedding variant (the study's
independent variable). Only ``linear`` exists today; conv variants register here
later without touching the encoder.
"""

from __future__ import annotations

from .models.patch_embed import LinearPatchEmbed
from .models.vit import ViT

_SIZES = {
    "tiny": dict(embed_dim=192, depth=12, num_heads=3, mlp_ratio=4.0),
    "small": dict(embed_dim=384, depth=12, num_heads=6, mlp_ratio=4.0),
}

_PATCH_EMBEDS = {
    "linear": LinearPatchEmbed,
}


def build_patch_embed(name: str, *, img_size: int, patch_size: int, in_chans: int, embed_dim: int):
    if name not in _PATCH_EMBEDS:
        raise ValueError(f"unknown patch_embed type {name!r}; have {list(_PATCH_EMBEDS)}")
    return _PATCH_EMBEDS[name](
        img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim
    )


def build_model(cfg: dict, num_classes: int) -> ViT:
    m = cfg["model"]
    size = m.get("size", "tiny")
    if size not in _SIZES:
        raise ValueError(f"unknown size {size!r}; have {list(_SIZES)}")
    size_args = _SIZES[size]
    embed_dim = size_args["embed_dim"]

    img_size = cfg["data"]["img_size"]
    patch_size = m["patch_size"]
    in_chans = m.get("in_chans", 3)

    patch_embed = build_patch_embed(
        m.get("patch_embed", "linear"),
        img_size=img_size,
        patch_size=patch_size,
        in_chans=in_chans,
        embed_dim=embed_dim,
    )
    return ViT(
        img_size=img_size,
        patch_size=patch_size,
        in_chans=in_chans,
        num_classes=num_classes,
        drop_path_rate=m.get("drop_path_rate", 0.0),
        drop_rate=m.get("drop_rate", 0.0),
        patch_embed=patch_embed,
        **size_args,
    )
