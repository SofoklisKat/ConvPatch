# ConvPatch Research and Experiment Plan

## Goal

Study whether replacing the standard **linear ViT patch embedding** with **convolutional patch embeddings** improves vision transformers when the rest of the model is held fixed.

The core rule is:

> Change only the patch-embedding stem. Keep token count, embedding dimension, Transformer encoder, and training recipe the same.

This makes the comparison about the **projection mechanism** itself:

- Linear patchify: one `P × P`, stride-`P` projection per patch.
- ConvPatch: local convolutional projection before producing the same token sequence.

---

## Research Question

At matched token count and training setup, do convolutional patch embeddings improve:

1. Classification accuracy
2. Training stability and convergence
3. Data efficiency
4. Robustness to corruptions
5. Accuracy/cost trade-off

---

## Hypothesis

Convolutional patch embeddings should produce better visual tokens because they add:

- local spatial bias
- progressive feature extraction
- optional overlap between neighboring regions
- normalization and nonlinearity before attention

Expected result: conv stems improve accuracy and convergence, especially on smaller-data settings.

---

## Patch-Embedding Variants

All variants output the same shape:

```text
image (B, C, H, W) -> tokens (B, N, D)
```

| ID | Variant | Config value | Description |
|----|---------|--------------|-------------|
| A | Linear baseline | `linear` | Standard ViT patchify: `Conv k=P, s=P` |
| B | Conv stem | `conv_stem` | Stacked stride-2 `3×3` convs |
| C | Overlapping | `overlapping` | Kernel larger than stride, patches overlap |
| D | Hierarchical | `hierarchical` | Downsample + refine at each stage |
| E | Depthwise-separable | `dwsep_conv_stem` | Efficient conv-stem variant |

Token-count constraint:

| Dataset | Image size | Patch size | Tokens |
|---------|------------|------------|--------|
| CIFAR-10 | `32×32` | `P=4` | `64` |
| CIFAR-100 | `32×32` | `P=4` | `64` |
| ImageNet-100 | `224×224` | `P=16` | `196` |

---

## Experiment Phases

### Phase 1 — CIFAR-10 Pilot

Purpose: verify the full pipeline and get the first linear-vs-conv signal.

Run all five variants with ViT-Tiny:

```text
cifar10_vit_tiny_linear
cifar10_vit_tiny_conv_stem
cifar10_vit_tiny_overlapping
cifar10_vit_tiny_hierarchical
cifar10_vit_tiny_dwsep_conv_stem
```

Current status:

- Linear baseline finished: **93.81% top-1**, 200 epochs.
- Conv-stem run is now in progress.

Output:

- Table: best top-1 per variant
- Plot: validation accuracy over epochs
- Check if conv variants are competitive before scaling up

---

### Phase 2 — CIFAR-100 Main Study

Purpose: primary controlled benchmark. CIFAR-100 is harder than CIFAR-10 and more useful for the paper.

Run:

```text
5 patch variants × 3 seeds × ViT-Tiny
```

Primary comparison:

```text
linear vs conv_stem
```

Main metric:

- top-1 accuracy, mean ± std over seeds

Secondary metrics:

- top-5 accuracy
- convergence speed
- train loss curve
- epoch time

---

### Phase 3 — Model Size Check

Purpose: test whether the result holds beyond ViT-Tiny.

Run the strongest variants on ViT-Small:

```text
linear
conv_stem
best_non_linear_variant_from_phase_2
```

Dataset:

- CIFAR-100 first
- ImageNet-100 later if compute allows

---

### Phase 4 — Data Efficiency

Purpose: test whether conv stems help more when less data is available.

Run on CIFAR-100 subsets:

```text
10% training data
25% training data
100% training data
```

Compare:

```text
linear vs conv_stem vs best variant
```

Expected signal:

- conv advantage should be larger at 10% and 25%.

---

### Phase 5 — Ablations

Purpose: explain why conv patch embedding helps.

Minimum required ablations:

| Ablation | Question |
|----------|----------|
| Position embedding removed | Does conv stem encode enough spatial structure? |
| Param-matched linear | Are gains just from more parameters? |
| No activation | Is nonlinearity in the stem important? |
| No normalization | Is BatchNorm part of the gain? |

Most important:

```text
A4: position embedding
A6: parameter-matched linear baseline
```

---

### Phase 6 — Robustness and Cost

Purpose: check whether gains are practical.

Measure:

- CIFAR-10-C / CIFAR-100-C corruption robustness
- throughput images/sec
- peak GPU memory
- parameter count
- FLOPs

Expected signal:

- conv stem improves robustness with small overhead.

---

## Paper Structure

### 1. Introduction

Motivate the problem: ViT uses a simple linear patchify stem, but CNN-style early processing may produce better tokens.

### 2. Related Work

Discuss:

- ViT
- DeiT
- Early Convolutions Help Transformers See Better
- CvT
- Swin / PVT / hierarchical ViTs

Positioning:

> Prior work shows conv stems help, but ConvPatch isolates the patch-embedding module under constant token count and fixed encoder.

### 3. Method

Define:

- linear baseline
- conv-stem variants
- constant-token-count rule
- fixed ViT encoder

### 4. Experiments

Describe:

- datasets
- training recipe
- model sizes
- metrics
- seeds

### 5. Results

Report:

- main accuracy table
- convergence curves
- low-data results
- ablations
- robustness/cost table

### 6. Discussion

Interpret:

- when conv stems help
- whether gains come from locality, overlap, or regularization
- limitations of small-scale experiments

### 7. Conclusion

Summarize the controlled finding and future work.

---

## Minimum Publishable Result

The smallest useful version of this study is:

```text
CIFAR-100
ViT-Tiny
5 patch variants
3 seeds
A4 position-embedding ablation
A6 param-matched linear control
```

This is enough to support the main claim if the conv variants consistently beat the linear baseline.

---

## Immediate Next Steps

1. Let CIFAR-10 sweep finish.
2. Summarize CIFAR-10 results with `scripts/summarize_runs.py`.
3. Add CIFAR-100 configs for all variants.
4. Run CIFAR-100 ViT-Tiny sweep.
5. Add aggregation plots/tables for the paper.
6. Implement A4 and A6 ablations.

