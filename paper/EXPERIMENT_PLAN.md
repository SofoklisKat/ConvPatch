# ConvPatch — Experiment Plan

> **Study type:** Controlled isolation study  
> **Compute:** Single GPU (GTX 1050 Ti, 4 GB)  
> **Principle:** Change *only* the patch-embedding module; hold encoder, token count, and training recipe fixed.

This document turns `DESIGN.md` into an executable experiment schedule: what to run, in what order, with what metrics, and how results map to paper claims.

---

## 1. Claims the paper must support

| Claim ID | Statement | Primary evidence |
|----------|-----------|------------------|
| C1 | Convolutional patch embeddings outperform linear patchify at **matched token count** | Main table: top-1 on CIFAR-10/100 and ImageNet-100 |
| C2 | Gains are **not** explained by extra parameters alone | A6 param-matched wide-linear control |
| C3 | Conv stems improve **optimization** (faster convergence, LR tolerance) | Convergence curves + LR sweep |
| C4 | Conv stems improve **data efficiency** | 10% / 25% low-data subsets |
| C5 | Conv stems improve **robustness** | CIFAR-10/100-C mCE |
| C6 | Cost overhead is **small** | Throughput, peak memory, FLOPs |

---

## 2. Independent variable: patch-embedding variants

All variants implement `image → (B, N, D)` with **identical** `N` and `D`. Config key: `model.patch_embed`.

| ID | Name | Config value | Description | Params (ViT-Ti, CIFAR P=4) |
|----|------|--------------|-------------|----------------------------|
| A | Linear | `linear` | Standard ViT: stride-P, P×P conv | 5.36M (baseline) |
| B | Conv stem | `conv_stem` | log₂(P) stride-2 3×3 blocks + 1×1 | 5.56M |
| C | Overlapping | `overlapping` | kernel=2P, stride=P | 5.39M |
| D | Hierarchical | `hierarchical` | Downsample + refine per stage | 5.97M |
| E | DW-sep stem | `dwsep_conv_stem` | Depthwise-separable stem | 5.41M |

**Token invariant:** total stem stride = `P` → `N = (H/P)²`.

| Dataset | Resolution | P | N tokens | Classes |
|---------|------------|---|----------|---------|
| CIFAR-10 | 32×32 | 4 | 64 | 10 |
| CIFAR-100 | 32×32 | 4 | 64 | 100 |
| ImageNet-100 | 224×224 | 16 | 196 | 100 |

---

## 3. Held constant (controls)

Across **all** arms in a given comparison:

- **Encoder:** ViT-Tiny or ViT-Small (depth, width, heads, MLP ratio, drop-path)
- **Token count N** and **embed dim D**
- **Optimizer:** AdamW, same LR schedule (warmup-cosine)
- **Augmentation:** RandAugment + RandomCrop(pad=4) + flip; mixup/cutmix (α=0.8/1.0)
- **Regularization:** weight decay 0.05, label smoothing 0.1, grad clip 1.0
- **AMP:** enabled on GPU
- **Seeds:** {0, 1, 2} for main results

Recipe is **not** tuned per variant. LR sensitivity is reported separately (§5.3).

---

## 4. Experiment phases

### Phase 0 — Sanity & baseline (Week 1) ✅ mostly done

| Run ID | Config | Purpose | Status |
|--------|--------|---------|--------|
| P0-1 | `cifar10_vit_tiny_linear` | Verify pipeline, establish linear baseline | Config ready |
| P0-2 | `cifar10_vit_tiny_conv_stem` | Quick conv vs linear signal | Config ready |
| P0-3 | All 5 CIFAR-10 configs | Forward + param parity | Smoke-tested |

**Gate:** linear baseline reaches competitive CIFAR-10 accuracy (target ≥ 75% top-1 @ 200 ep for ViT-Ti; literature ~78–85% with strong aug).

---

### Phase 1 — Main factorial (Weeks 2–4)

**Goal:** Answer RQ1 (accuracy) with statistical rigor.

```
Variants (5) × Sizes (2) × Datasets (2) × Seeds (3) = 60 runs
```

| Dataset | Model | Variants | Epochs | Est. time/run (1050 Ti) |
|---------|-------|----------|--------|--------------------------|
| CIFAR-10 | ViT-Ti | A–E | 200 | ~2.5 h |
| CIFAR-100 | ViT-Ti | A–E | 200 | ~2.5 h |
| CIFAR-100 | ViT-S | A–E | 200 | ~5 h |
| ImageNet-100 | ViT-Ti | A–E | 100 | ~8 h |
| ImageNet-100 | ViT-S | A–E | 100 | ~15 h |

**Priority order** (if compute-limited):

1. CIFAR-100, ViT-Ti, A vs B (linear vs conv_stem) × 3 seeds — **minimum viable paper**
2. Add C, D, E on CIFAR-100 ViT-Ti
3. CIFAR-100 ViT-S
4. ImageNet-100 ViT-Ti

**Outputs:**

- Table 1: Top-1 / Top-5 (mean ± std over seeds)
- Figure 1: Bar chart or dot plot, variant × dataset
- Figure 2: Training curves (loss + val acc), linear vs best conv

**Run naming convention:**

```
{dataset}_{size}_{variant}_seed{seed}
# e.g. cifar100_tiny_conv_stem_seed1
```

---

### Phase 2 — Low-data regime (Week 4)

**Goal:** Answer RQ3 (inductive bias / data efficiency).

| Run ID | Data fraction | Variants | Seeds |
|--------|---------------|----------|-------|
| LD-10 | 10% train | A, B (+ best from Phase 1) | 3 |
| LD-25 | 25% train | A, B (+ best) | 3 |

**Protocol:**

- Stratified random subset of training set; **fixed indices per seed** (saved to `data/subsets/`)
- Identical recipe otherwise
- Report: top-1 vs data fraction curve

**Expected signal:** larger conv−linear gap at 10% than at 100%.

---

### Phase 3 — Optimization & stability (Week 5)

**Goal:** Answer RQ2.

| Experiment | Variants | Sweeps |
|------------|----------|--------|
| 3.1 Convergence | A, B | Plot val acc vs epoch; epochs-to-90%-best |
| 3.2 LR sensitivity | A, B | LR ∈ {3e-4, 1e-3, 3e-3, 1e-2} |
| 3.3 Warmup sensitivity | A, B | warmup ∈ {0, 5, 20} epochs |
| 3.4 Optimizer | A, B | AdamW vs SGD (momentum 0.9) |

**Dataset:** CIFAR-100, ViT-Ti (primary); repeat key finding on ImageNet-100 if budget allows.

**Outputs:**

- Figure 3: LR sensitivity heatmap or line plot
- Figure 4: Convergence comparison (median + IQR over seeds)

---

### Phase 4 — Ablations (Weeks 5–6)

**Goal:** Isolate *why* conv helps (DESIGN.md §7).

| Ablation | Factor | Levels | Load-bearing? |
|----------|--------|--------|---------------|
| A1 | Stem depth | 1 / 2 / 4 stride-2 blocks | Medium |
| A2 | Overlap | kernel=stride vs kernel>stride | Medium |
| A3 | Normalization | BN vs LN vs none | Medium |
| A4 | Position embedding | learned / sincos / **none** | **Yes** |
| A5 | Nonlinearity | GELU vs identity | Medium |
| A6 | Param-matched linear | widen linear to match B params | **Yes** |

**Minimum for publication:** A4 + A6 on CIFAR-100 ViT-Ti, seeds=3.

**A4 hypothesis:** conv stem reduces reliance on explicit positional encoding.

**A6 hypothesis:** conv gains persist even when linear arm has equal parameter count.

---

### Phase 5 — Robustness & efficiency (Week 6)

**Goal:** Answer RQ4 and RQ5.

| Metric | Protocol |
|--------|----------|
| CIFAR-10-C / CIFAR-100-C | Evaluate best checkpoint; report mCE (lower is better) |
| Throughput | img/s @ batch 128, inference mode, 1000 iterations |
| Peak memory | `torch.cuda.max_memory_allocated` during train step |
| FLOPs | `fvcore` or manual count for patch-embed + encoder |

**Outputs:**

- Table 2: mCE by corruption type (aggregate + mean)
- Table 3: Params, FLOPs, throughput, memory

---

### Phase 6 — Analysis (optional, Week 7)

Interpretability (not required for first submission, strengthens paper):

- Effective receptive field of patch-embed output tokens
- Token cosine similarity (intra- vs inter-image)
- Attention distance statistics (early vs late layers)
- Fourier / frequency response of stem vs linear

---

## 5. Metrics & logging

### Per-run artifacts (`runs/<run_name>_seed<N>/`)

| File | Contents |
|------|----------|
| `config.json` | Full reproducibility snapshot |
| `log.csv` | epoch, train_loss, lr, top1, top5, ms_per_iter, epoch_s |
| `best.pt` / `last.pt` | Checkpoints |
| `summary.json` | best_top1, params, tokens, epochs |

### Aggregated results (`results/`)

| File | Contents |
|------|----------|
| `main_results.csv` | variant × dataset × size × seed → top1, top5 |
| `main_results_agg.csv` | mean ± std per (variant, dataset, size) |
| `ablations.csv` | ablation factor × level → top1 |
| `efficiency.csv` | variant → params, flops, throughput, memory |

---

## 6. Statistical reporting

- Report **mean ± std** over 3 seeds for all main numbers.
- Use paired comparison: conv_stem vs linear per seed where possible.
- Significance: bootstrap 95% CI on mean top-1 difference (no p-hacking across 5 variants; pre-register primary comparison: **B vs A**).
- Primary endpoint: **top-1 accuracy on CIFAR-100, ViT-Ti, seed-averaged**.

---

## 7. Compute budget estimate

| Phase | Runs | Est. GPU-hours |
|-------|------|----------------|
| Phase 1 (full 60) | 60 | ~250 h |
| Phase 1 (MVP: CIFAR-100 Ti, A–E) | 15 | ~40 h |
| Phase 2 low-data | 12 | ~30 h |
| Phase 3 optimization | ~20 | ~50 h |
| Phase 4 ablations (A4+A6) | 12 | ~30 h |
| Phase 5 robustness | 10 | ~10 h (eval only) |

**MVP path (~80 GPU-hours):** Phase 1 subset (CIFAR-100 ViT-Ti, all variants) + A4 + A6 + convergence plots.

---

## 8. Paper figure / table mapping

| Paper asset | Source experiment | Status |
|-------------|-------------------|--------|
| Fig. 1 — Method overview | Architecture diagram | Draft in paper |
| Fig. 2 — Design space | Variants A–E schematic | Draft in paper |
| Table 1 — Main results | Phase 1 | Pending |
| Fig. 3 — Convergence | Phase 3.1 | Pending |
| Fig. 4 — LR sensitivity | Phase 3.2 | Pending |
| Table 2 — Low-data | Phase 2 | Pending |
| Table 3 — Ablations A4, A6 | Phase 4 | Pending |
| Table 4 — Robustness mCE | Phase 5 | Pending |
| Table 5 — Efficiency | Phase 5 | Pending |

---

## 9. Immediate next actions

1. **Train linear baseline** on CIFAR-10 (200 ep) — reference checkpoint.
2. **Train conv_stem** on CIFAR-10 (200 ep) — first comparison.
3. Add **CIFAR-100 configs** (copy CIFAR-10, change `dataset` + `num_classes`).
4. Implement **subset sampler** for low-data (Phase 2).
5. Implement **A6 wide-linear** patch embed.
6. Add **`results/aggregate.py`** script to build tables from `runs/`.

---

## 10. Risk register

| Risk | Mitigation |
|------|------------|
| Conv gains only on CIFAR, not ImageNet-100 | Report both; frame as small-scale controlled study |
| Hierarchical (D) wins on params unfairly | Report per-variant params; include A6 |
| 200 epochs too slow | Pilot at 100 ep; early-stop if curves plateau |
| CIFAR-10 vs CIFAR-100 confusion | CIFAR-10 = dev/pilot; CIFAR-100 = main table |
