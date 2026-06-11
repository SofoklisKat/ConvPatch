# ConvPatch: Convolutional Patch Embedding for Vision Encoders

> **Type:** Controlled isolation study (rigor-focused)
> **Compute regime:** Small (single-GPU) — CIFAR-100 and ImageNet-100, ViT-Ti / ViT-S
> **One-line thesis:** Holding the Transformer encoder and training recipe fixed, replacing the *linear* patch projection with a small *convolutional stem* improves accuracy, optimization stability, and data efficiency at a matched parameter / FLOP / token budget.

---

## 1. Motivation & hypothesis

Standard Vision Transformers (ViT) "patchify" an image by slicing it into non-overlapping squares (e.g. 16×16), flattening each, and applying a **single linear projection**. That projection is exactly a stride-`P` convolution with kernel `P`: *no overlap, no hierarchy, no locality beyond the patch boundary.*

**Central hypothesis (H0):** A small **convolutional stem** (overlapping, multi-layer, with normalization and nonlinearity) produces tokens that are easier for the Transformer to optimize — yielding higher accuracy, faster/more stable convergence, and better data efficiency — **at matched token count, parameter count, and FLOPs.**

This connects to prior observations (Xiao et al., *Early Convolutions Help Transformers See Better*; CvT; LeViT; PVTv2). **Our contribution is methodological rigor, not a new SOTA number:** a clean "change one thing" study that holds the encoder and recipe fixed and varies *only* the patch-embedding module across a principled design space, with the capacity confound explicitly controlled.

---

## 2. Research questions

```mermaid
mindmap
  root((Conv Patch<br/>Embedding))
    RQ1 Accuracy
      Matched params/FLOPs
      Matched token count
    RQ2 Optimization
      Convergence speed
      LR/warmup sensitivity
      Stability without heavy aug
    RQ3 Inductive bias
      Data efficiency low-data
      Is position emb still needed?
    RQ4 Robustness
      CIFAR-100-C corruptions
      Texture vs shape bias
    RQ5 Cost
      Throughput img/s
      Peak memory
```

> Model-size / full-ImageNet **scaling** is explicitly **out of scope** for this small-compute study and is listed under Future Work (§11).

---

## 3. Design space (the independent variable)

We vary only the patch-embedding module `f: image → tokens`. Every variant emits the **same number of tokens `N`** and the **same embedding dim `D`**, so the downstream encoder is byte-for-byte identical across arms.

```mermaid
flowchart TD
    IMG["Input image<br/>H x W x 3"] --> PE{Patch embedding<br/>variant}

    PE --> A["A. Linear baseline (ViT)<br/>Conv k=P s=P, no overlap"]
    PE --> B["B. Conv stem<br/>stack of 3x3 conv + norm + act<br/>total stride = P"]
    PE --> C["C. Overlapping patch<br/>kernel > stride (CvT/PVTv2 style)"]
    PE --> D["D. Hierarchical stem<br/>progressive downsample to /P"]
    PE --> E["E. Depthwise-separable stem<br/>param-efficient"]

    A --> TOK["N tokens x D"]
    B --> TOK
    C --> TOK
    D --> TOK
    E --> TOK

    TOK --> ENC["FIXED Transformer encoder<br/>L layers, identical across all arms"]
    ENC --> HEAD["Classification head"]
```

### What is held constant vs varied

```mermaid
flowchart LR
    subgraph Controls["Held constant (controls)"]
        T1["Token count N"]
        T2["Embed dim D"]
        T3["Encoder depth / width / heads"]
        T4["Training recipe<br/>optimizer, epochs, aug, schedule"]
    end
    subgraph Matched["Matched per comparison"]
        M1["Params"]
        M2["FLOPs"]
    end
    subgraph IV["Independent variable"]
        V1["Patch-embedding structure"]
    end
    IV --> OUT["Dependent variables:<br/>acc, convergence, robustness, throughput"]
    Controls --> OUT
    Matched --> OUT
```

---

## 4. Architecture detail: linear vs conv stem

```mermaid
flowchart TB
    subgraph Linear["A. Linear patchify (baseline)"]
        L1["Conv2d k=P s=P<br/>3 -> D"] --> L2["Flatten -> N x D"] --> L3["+ pos embed"]
    end
    subgraph Conv["B. Convolutional stem (proposed)"]
        C1["3x3 s2, 3->D/8, Norm, GELU"] --> C2["3x3 s2, D/8->D/4, Norm, GELU"]
        C2 --> C3["3x3 s2, D/4->D/2, Norm, GELU"] --> C4["3x3 s2, D/2->D, Norm, GELU"]
        C4 --> C5["1x1 conv -> D"] --> C6["Flatten -> N x D"] --> C7["+ pos embed (optional)"]
    end
```

**Token-count invariant:** total stem stride must equal `P`. For CIFAR-100 at 32×32 with `P=4` (→ `N=64` tokens), two stride-2 blocks suffice; for ImageNet-100 at 224×224 with `P=16` (→ `N=196` tokens), four stride-2 blocks. The final `1×1` conv guarantees output channels = `D`, matching the linear arm exactly.

---

## 5. Experimental matrix (small-compute factorial)

```mermaid
flowchart LR
    subgraph Variants["Patch variants"]
        v["A / B / C / D / E"]
    end
    subgraph Sizes["Encoder sizes"]
        s["ViT-Ti / ViT-S"]
    end
    subgraph Data["Datasets / regimes"]
        d["CIFAR-100 (P=4)<br/>ImageNet-100 (P=16)<br/>+ 10% / 25% low-data subsets"]
    end
    Variants --> RUN["Factorial sweep<br/>seeds x3"]
    Sizes --> RUN
    Data --> RUN
    RUN --> EVAL["Evaluation suite (§6)"]
```

**Run budget estimate:** 5 variants × 2 sizes × 2 datasets × 3 seeds = 60 main runs, plus low-data and ablation runs. Each fits a single consumer GPU at these resolutions/token counts; CIFAR-100 runs are minutes-to-an-hour, ImageNet-100 a few hours each.

---

## 6. Evaluation pipeline

```mermaid
flowchart TD
    CKPT["Trained checkpoint"] --> ACC["Top-1 / Top-5 accuracy"]
    CKPT --> CONV["Convergence<br/>epochs-to-X% acc, loss curves"]
    CKPT --> ROB["Robustness<br/>CIFAR-100-C mCE, low-data acc"]
    CKPT --> EFF["Efficiency<br/>throughput img/s, peak mem, FLOPs, params"]
    CKPT --> ANA["Analysis<br/>attention maps, effective receptive field,<br/>token similarity, Fourier/freq response"]
    ACC --> REP["Paper tables + plots"]
    CONV --> REP
    ROB --> REP
    EFF --> REP
    ANA --> REP
```

---

## 7. Ablations — isolate *why* it helps

```mermaid
flowchart TD
    Q["Why does the conv stem help?"] --> A1["Stem depth<br/>1 vs 2 vs 4 conv layers"]
    Q --> A2["Overlap<br/>kernel=stride vs kernel>stride"]
    Q --> A3["Normalization<br/>BN vs LN vs none"]
    Q --> A4["Position embedding<br/>learned vs sincos vs NONE"]
    Q --> A5["Nonlinearity<br/>with vs without activations"]
    Q --> A6["Param-matched linear control<br/>wider/factorized linear at same params"]
```

**A6 and A4 are the load-bearing controls.** A6 rules out "it just has more parameters." A4 tests the common claim that a conv stem injects enough positional information to make explicit position embeddings redundant.

---

## 8. Hypotheses → predicted signals

| RQ  | Hypothesis | Predicted signal |
|-----|-----------|------------------|
| RQ1 | Conv > linear at matched budget | +1–3% top-1, larger gap on smaller data |
| RQ2 | Conv trains more stably | Tolerates higher LR, needs less warmup, lower loss variance |
| RQ3 | Conv adds locality bias | Bigger gains in 10%/25% low-data; pos-emb less critical (A4) |
| RQ4 | Conv more robust | Lower mCE on CIFAR-100-C |
| RQ5 | Conv stem is cheap | < 5% throughput cost vs linear |

---

## 9. Project phases

```mermaid
gantt
    title ConvPatch research timeline (small-compute)
    dateFormat YYYY-MM-DD
    section Setup
    Repo, data pipeline, configs        :a1, 2026-06-10, 7d
    Reproduce ViT linear baseline       :a2, after a1, 5d
    section Core experiments
    Implement patch variants A-E        :b1, after a2, 7d
    CIFAR-100 sweeps                     :b2, after b1, 10d
    ImageNet-100 sweeps                  :b3, after b2, 14d
    section Analysis
    Ablations (A1-A6)                    :c1, after b2, 12d
    Robustness + efficiency             :c2, after b3, 7d
    Interpretability analysis           :c3, after b3, 7d
    section Writeup
    Draft paper + figures               :d1, after c3, 14d
```

---

## 10. Threats to validity & mitigations

| Threat | Mitigation |
|--------|-----------|
| Capacity confound (conv just adds params) | Strict param/FLOP matching + A6 wide-linear control |
| Recipe favoritism (recipe tuned for conv) | Identical recipe across arms; report LR-sweep *sensitivity*, not one tuned point |
| Token-count mismatch | Enforce stem stride = `P` so `N` is identical across arms |
| Seed noise (amplified at small scale) | 3 seeds; report mean ± std, especially for low-data |
| Single-dataset cherry-pick | Span CIFAR-100 → ImageNet-100 + low-data subsets |
| Position-embedding interaction | A4 ablation explicitly crosses stem × pos-emb |

---

## 11. Out of scope / future work
- Full ImageNet-1k and ViT-B scaling laws (needs multi-GPU).
- Self-supervised pretraining (MAE/DINO) with conv stems.
- Dense prediction (detection/segmentation) transfer.
- Hybrid conv-attention encoders (we keep the encoder pure to isolate the stem).

---

## 12. Deliverables
- Reproducible config-driven training code, fixed seeds, logged runs.
- Main results table (variant × size × dataset) with mean ± std.
- Ablation tables A1–A6.
- Convergence, robustness, efficiency, and interpretability figures.
- Paper draft positioning this as a controlled isolation study.
