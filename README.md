# ConvPatch

A **controlled isolation study** of convolutional vs. linear **patch embedding** in Vision Transformers.

Standard ViTs patchify an image with a single stride-`P`, `P×P` convolution — equivalent to flattening non-overlapping patches and applying one linear projection. **ConvPatch** asks:

> At **fixed token count**, **fixed Transformer encoder**, and **fixed training recipe**, does the *structure* of the patch-embedding stem affect accuracy — and which conv design helps?

This is a **change-one-thing** experiment: only the patch-embedding module varies; everything downstream is identical.

---

## Research question

Does replacing linear patchify with a convolutional stem improve ViT performance when capacity and sequence length are controlled?

We test five stems (A–E) on CIFAR-10/100 with ViT-Tiny, measuring top-1 accuracy, convergence, and (planned) data efficiency and robustness.

---

## Key finding (Phase 1, seed 0)

| Patch embed | CIFAR-10 | CIFAR-100 | Params (M) |
|-------------|----------:|----------:|-----------:|
| **A: linear** (baseline) | **93.71** | 74.10 | 5.38 |
| B: conv_stem | 93.40 | 72.87 | 5.58 |
| C: overlapping | 93.69 | 73.89 | 5.41 |
| **D: hierarchical** | 93.27 | **75.84** | 5.99 |
| E: dwsep_conv_stem | 89.20 | 67.71 | 5.43 |

**Takeaway:** Conv stems are **not universally better**. On harder **CIFAR-100**, hierarchical patch embedding beats linear by **+1.74 pp**; on near-saturated **CIFAR-10**, linear is marginally best. Depthwise-separable stems underperform on both.

Full analysis: [`paper/RESULTS.md`](paper/RESULTS.md) · CSV: [`paper/results/main_results_seed0.csv`](paper/results/main_results_seed0.csv)

---

## Method

### What is patch embedding?

```
image (H×W×3)  →  patch embedding  →  tokens (N×D)  →  ViT encoder  →  logits
```

- **N** = `(H/P)²` tokens (held **constant** across all arms)
- **D** = embedding dim (192 for ViT-Tiny)
- Only the orange box above is swapped in this study

Visual guide: [`paper/figures/VISUAL_GUIDE.md`](paper/figures/VISUAL_GUIDE.md)

### Patch-embedding variants (independent variable)

| ID | Name | Config key | Description |
|----|------|------------|-------------|
| A | Linear | `linear` | Standard ViT: `Conv k=P, s=P` |
| B | Conv stem | `conv_stem` | Stacked stride-2 `3×3` convs (Xiao-style) |
| C | Overlapping | `overlapping` | Kernel `2P`, stride `P` (CvT-style) |
| D | Hierarchical | `hierarchical` | Downsample + refine per stage |
| E | DW-separable | `dwsep_conv_stem` | Param-efficient conv stem |

### Controls (held constant)

- ViT-Tiny encoder (12 layers, 192 dim, 3 heads)
- Token count **N = 64** (32×32 images, P=4)
- Training recipe: AdamW, warmup-cosine, RandAugment, mixup/cutmix, label smoothing
- 200 epochs, batch 128, AMP on GPU

### Positioning vs. prior work

Closest prior: [Xiao et al., *Early Convolutions Help Transformers See Better*](https://arxiv.org/abs/2106.14881) (NeurIPS 2021).

**Our wedge:** constant token count (no encoder block removal), five stem designs under one recipe, small-compute reproducibility. See [`DESIGN.md`](DESIGN.md) and [`paper/ANALYSIS_GOAL.md`](paper/ANALYSIS_GOAL.md).

---

## Study status

| Phase | Description | Status |
|-------|-------------|--------|
| 0 | Pipeline + smoke tests | done |
| 1 | CIFAR-10/100, 5 variants, seed 0 | **done** |
| 1b | CIFAR-100 seeds 1–2 | pending |
| 2 | Low-data (10%, 25%) | pending |
| 3 | LR / convergence sweeps | pending |
| 4 | Ablations (pos-emb, param-matched linear) | pending |
| 5 | Robustness (CIFAR-C) + efficiency | pending |

| Artifact | Location |
|----------|----------|
| Research design | [`DESIGN.md`](DESIGN.md) |
| Experiment plan | [`paper/EXPERIMENT_PLAN.md`](paper/EXPERIMENT_PLAN.md) |
| Research plan | [`paper/RESEARCH_PLAN.md`](paper/RESEARCH_PLAN.md) |
| Results (seed 0) | [`paper/RESULTS.md`](paper/RESULTS.md) |
| Paper draft (LaTeX) | [`paper/main.tex`](paper/main.tex) |
| Remote GPU guide | [`REMOTE_GPU.md`](REMOTE_GPU.md) |

---

## Quick start

### Setup

```bash
git clone git@github.com:SofoklisKat/ConvPatch.git
cd ConvPatch
./scripts/setup_remote.sh
# CUDA 12.4 wheels: CUDA_INDEX=https://download.pytorch.org/whl/cu124 ./scripts/setup_remote.sh
```

Or manually:

```bash
uv venv --python 3.12 .venv
uv pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu124
uv pip install -e .
```

### Run experiments

```bash
# CIFAR-100 sweep (5 variants, seed 0)
./scripts/run_cifar100_experiments.sh

# CIFAR-100, seeds 0+1+2 (paper protocol)
DEVICE=cuda:0 ./scripts/run_cifar100_all_seeds.sh

# CIFAR-10 sweep
./scripts/run_cifar10_experiments.sh

# Single run
.venv/bin/python -m convpatch.train --config configs/cifar100_vit_tiny_linear.yaml

# Override config from CLI
.venv/bin/python -m convpatch.train \
  --config configs/cifar100_vit_tiny_hierarchical.yaml \
  --set train.device=cuda:0 seed=1 train.epochs=200
```

### Summarize results

```bash
.venv/bin/python scripts/summarize_runs.py
```

### Background run

```bash
nohup ./scripts/run_cifar100_all_seeds.sh > logs/cifar100_sweep.log 2>&1 &
tail -f logs/cifar100_sweep.log
```

---

## Requirements

- Python ≥ 3.10
- NVIDIA GPU recommended (CUDA 12.x; use matching PyTorch wheel: `cu121`, `cu124`, etc.)
- CPU works (disable AMP; much slower)

---

## Project layout

```
configs/                    # YAML configs per dataset × variant
  cifar10_vit_tiny_*.yaml
  cifar100_vit_tiny_*.yaml
src/convpatch/
  models/patch_embed.py     # Variants A–E (independent variable)
  models/vit.py             # Fixed ViT encoder
  train.py                  # Config-driven training entry point
  data.py                   # CIFAR-10/100 loaders
  build.py                  # Config → model
scripts/
  setup_remote.sh           # Venv + CUDA install on new machine
  run_experiments.sh        # Sweep runner (DATASET=cifar10|cifar100)
  run_cifar100_experiments.sh
  run_cifar100_all_seeds.sh
  summarize_runs.py         # Results table from runs/
paper/
  RESULTS.md                # Written results + interpretation
  main.tex                  # Paper draft
  EXPERIMENT_PLAN.md        # Full experiment schedule
  figures/                  # Concept diagrams
runs/                       # Training outputs (gitignored)
  <run_name>_seed<N>/
    log.csv, best.pt, summary.json
DESIGN.md                   # Research design (mermaid diagrams)
REMOTE_GPU.md               # rsync/scp + remote setup
```

---

## Reproducing published numbers

Phase 1 results (seed 0, 200 epochs):

```bash
DATASET=cifar100 SEED=0 EPOCHS=200 DEVICE=cuda:0 ./scripts/run_experiments.sh
```

Expected best top-1 (approximate): hierarchical **75.84%**, linear **74.10%** on CIFAR-100.

---

## Citation

```bibtex
@misc{convpatch2026,
  title={ConvPatch: A Controlled Study of Convolutional vs. Linear Patch Embedding in Vision Transformers},
  author={Katakis, Sofoklis},
  year={2026},
  url={https://github.com/SofoklisKat/ConvPatch}
}
```

## License

TBD.
