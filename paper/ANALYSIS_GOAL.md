# Literature analysis goal

## What is patch embedding?

The first step of a Vision Transformer: map an image `(H × W × 3)` to a token sequence `(N × D)` that the Transformer encoder consumes.

- **N** = number of tokens = `(H/P) × (W/P)` for patch size **P**
- **D** = embedding dimension (e.g. 192 for ViT-Tiny)

Everything after this (class token, positional embedding, attention blocks) is the **encoder** — held fixed in ConvPatch.

### Linear (standard ViT)

One convolution: kernel **P × P**, stride **P**. Equivalent to splitting the image into non-overlapping P×P patches and applying a single linear projection per patch. No overlap, no extra layers, no nonlinearity in the stem.

Example: CIFAR 32×32, P=4 → **64 tokens**.

### Convolutional alternatives

Replace that single large patchify conv with a small **conv stem** — stacked 3×3 convs, overlap, norm, activation — while keeping the **same N** by forcing total stride = P. The stem changes *how* local structure is extracted before attention; the encoder does not change.

ConvPatch compares five stems (linear, conv, overlapping, hierarchical, depthwise-separable) under matched token count and training recipe.

### Dummy figures (visual)

See **[figures/VISUAL_GUIDE.md](figures/VISUAL_GUIDE.md)** — all 6 images with short descriptions.

Regenerate: `python paper/figures/generate_dummies.py`

---

## Analysis goal

Map prior work on **ViT patch embedding** (linear vs. convolutional stems) and extract, for each paper:

- what was changed (stem only vs. full architecture)
- whether **token count**, **params/FLOPs**, and **training recipe** were held constant
- what evidence supports their claims (metrics, ablations, datasets)

**Outcome:** a comparison matrix that positions ConvPatch (constant-N, controlled factorial study) against existing work and drives the related-work section + experiment gaps (A4, A6, low-data, robustness).
