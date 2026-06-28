# ConvPatch — Experimental Results (seed 0)

> **Status:** Phase 1 complete for CIFAR-10 and CIFAR-100 (ViT-Tiny, 200 epochs, seed 0).  
> **Setup:** Identical training recipe across all arms; only `patch_embed` varies; N=64 tokens, P=4.

Machine-readable table: [`results/main_results_seed0.csv`](results/main_results_seed0.csv)

---

## 1. Main results

### CIFAR-100 (primary benchmark)

| Rank | Variant | Patch embed | Top-1 (%) | Δ vs linear | Params (M) |
|------|---------|-------------|----------:|------------:|-----------:|
| 1 | D | hierarchical | **75.84** | **+1.74** | 5.99 |
| 2 | A | linear | 74.10 | — | 5.38 |
| 3 | C | overlapping | 73.89 | −0.21 | 5.41 |
| 4 | B | conv_stem | 72.87 | −1.23 | 5.58 |
| 5 | E | dwsep_conv_stem | 67.71 | −6.39 | 5.43 |

**Best arm:** hierarchical (+1.74 pp over linear baseline).

### CIFAR-10 (pilot / sanity check)

| Rank | Variant | Patch embed | Top-1 (%) | Δ vs linear | Params (M) |
|------|---------|-------------|----------:|------------:|-----------:|
| 1 | A | linear | **93.71** | — | 5.36 |
| 2 | C | overlapping | 93.69 | −0.02 | 5.39 |
| 3 | B | conv_stem | 93.40 | −0.31 | 5.56 |
| 4 | D | hierarchical | 93.27 | −0.44 | 5.97 |
| 5 | E | dwsep_conv_stem | 89.20 | −4.51 | 5.41 |

**Best arm:** linear (conv variants within ~0.5 pp; dwsep clearly worse).

---

## 2. Combined summary table

| Patch embed | CIFAR-10 top-1 | CIFAR-100 top-1 | Params (M) |
|-------------|---------------:|----------------:|-----------:|
| A: linear | 93.71 | 74.10 | 5.36–5.38 |
| B: conv_stem | 93.40 | 72.87 | 5.56–5.58 |
| C: overlapping | 93.69 | 73.89 | 5.39–5.41 |
| D: hierarchical | 93.27 | **75.84** | 5.97–5.99 |
| E: dwsep_conv_stem | 89.20 | 67.71 | 5.41–5.43 |

All runs: ViT-Tiny, 32×32, patch size 4, 64 tokens, seed 0, 200 epochs, AMP, mixup/cutmix, RandAugment.

---

## 3. Key findings (seed 0)

1. **Task difficulty matters.** On near-saturated CIFAR-10 (~94% top-1), linear patchify is competitive or best. On harder CIFAR-100, a **hierarchical conv stem** outperforms linear by **+1.74 pp**.

2. **Not all conv stems help equally.** Hierarchical > overlapping ≈ linear > conv_stem >> dwsep on CIFAR-100. The benefit is **design-dependent**, not “any conv > linear.”

3. **Parameter count does not explain ranking.** Hierarchical has the most parameters and wins on CIFAR-100; dwsep is mid-sized but worst. Linear has the fewest params and wins on CIFAR-10.

4. **Depthwise-separable stem underperforms** on both datasets (−4.5 pp on CIFAR-10, −6.4 pp on CIFAR-100 vs linear). Likely too weak for from-scratch ViT training at this scale.

5. **Token count held constant** (N=64) across all arms — differences are attributable to stem structure, not sequence length.

---

## 4. Interpretation (draft for paper)

### CIFAR-100

The hierarchical stem (downsample + refine per stage) may extract richer local structure before global self-attention, which helps when the classification task is fine-grained (100 classes, similar low-level features). The simple conv stem (B) and overlapping single conv (C) show smaller or negative gaps vs linear at seed 0.

### CIFAR-10

High accuracy across arms suggests the benchmark is near saturation; inductive bias from conv stems has limited room to help. Linear remains a strong default for easy, small-scale classification.

### Relation to prior work

Xiao et al. report conv stems helping optimization and accuracy on ImageNet at matched FLOPs (with encoder block removed). Our setting holds **token count and encoder depth fixed** and finds that **hierarchical** conv embedding helps on CIFAR-100 but not uniformly across stem designs.

---

## 5. Statistical note

These are **single-seed (0)** results. For the paper, report mean ± std over seeds {0, 1, 2} on CIFAR-100 before drawing strong conclusions. The hierarchical lead (+1.74 pp) may shrink or grow with additional seeds.

**Pending:** seeds 1–2, low-data subsets, ablations A4/A6, robustness (CIFAR-C).

---

## 6. Run artifacts

| Run name | Log | Checkpoint |
|----------|-----|------------|
| `cifar10_vit_tiny_*_seed0` | `runs/*/log.csv` | `runs/*/best.pt` |
| `cifar100_vit_tiny_*_seed0` | `runs/*/log.csv` | `runs/*/best.pt` |

Regenerate summary:

```bash
.venv/bin/python scripts/summarize_runs.py
```

---

## 7. Suggested paper claims (seed 0, provisional)

- **Supported:** Patch-embedding *structure* affects accuracy at matched token count; effect is dataset- and design-dependent.
- **Supported:** Hierarchical conv stem improves CIFAR-100 ViT-Ti over linear (+1.74 pp, seed 0).
- **Not supported (yet):** Universal “conv > linear” claim; simple conv stem and dwsep do not beat linear on CIFAR-100.
- **Needs more seeds:** Whether hierarchical lead is statistically robust.
