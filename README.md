# ConvPatch

A controlled study of **convolutional vs. linear patch embedding** in vision encoders.

Standard Vision Transformers (ViT) "patchify" an image with a single stride-`P`,
`P×P` convolution — a linear projection of non-overlapping patches with no
locality beyond the patch boundary. **ConvPatch** asks: holding the Transformer
encoder and training recipe *fixed*, does replacing that linear stem with a small
**convolutional stem** improve accuracy, optimization stability, and data
efficiency at a matched parameter / FLOP / token budget?

This repo is a "change one thing" study: only the patch-embedding module varies;
the encoder is byte-for-byte identical across arms. See [`DESIGN.md`](DESIGN.md)
for the full research design (research questions, design space, ablations,
threats to validity) with diagrams.

> **Closest prior work:** Xiao et al., *Early Convolutions Help Transformers See
> Better*, NeurIPS 2021 ([arXiv:2106.14881](https://arxiv.org/abs/2106.14881)).
> Our contribution is rigor at small-compute scale: constant token count, a
> broader stem design space, and added data-efficiency / robustness /
> position-embedding-redundancy analyses. See `DESIGN.md` §1 for positioning.

## Status

| Component | State |
|---|---|
| Research design (`DESIGN.md`) | done |
| Standard ViT encoder (swappable patch embed) | done |
| Linear patch embedding (baseline) | done |
| Data pipeline (CIFAR-10/100, RandAugment, mixup/cutmix) | done |
| Config-driven training (AMP, warmup-cosine, ckpt/logging) | done |
| CIFAR-10 ViT-Tiny linear baseline | config ready |
| Conv patch-embedding variants (B–E) | planned |

## Requirements

- Python ≥ 3.10
- NVIDIA GPU + driver supporting CUDA 12.1 wheels (developed on a GTX 1050 Ti,
  driver 535 / CUDA 12.2). CPU-only also works (slower; disable AMP).

## Setup

Using [`uv`](https://github.com/astral-sh/uv) (recommended):

```bash
uv venv --python 3.12 .venv
uv pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu121
uv pip install -e .
```

For CPU-only, swap the index URL to `https://download.pytorch.org/whl/cpu`.

## Usage

Train the CIFAR-10 ViT-Tiny linear baseline:

```bash
.venv/bin/python -m convpatch.train --config configs/cifar10_vit_tiny_linear.yaml
```

Override any config field from the CLI with dotted keys:

```bash
.venv/bin/python -m convpatch.train \
  --config configs/cifar10_vit_tiny_linear.yaml \
  --set train.epochs=100 train.lr=0.0008 data.batch_size=256
```

Artifacts (per run, under `runs/<run_name>_seed<seed>/`):
`config.json`, `log.csv`, `last.pt`, `best.pt`, `summary.json`.

### Reference numbers

On a GTX 1050 Ti (AMP, batch 128, 64 tokens): ~342 ms/iter, ~142 s/epoch;
a full 200-epoch baseline takes ~8 hours.

## Project layout

```
configs/                 # YAML experiment configs
src/convpatch/
  data.py                # CIFAR-10/100 datasets + augmentation
  build.py               # config -> model (patch-embed type = study variable)
  engine.py              # train/eval epoch loops (AMP, mixup)
  train.py               # config-driven entry point
  utils.py               # seeding, metrics, scheduler, mixup/cutmix
  models/
    patch_embed.py       # LinearPatchEmbed (conv variants to follow)
    vit.py               # standard ViT with swappable patch embedding
DESIGN.md                # research design + diagrams
```

## License

TBD.
