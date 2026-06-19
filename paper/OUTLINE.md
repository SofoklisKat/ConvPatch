# Paper outline (working)

Quick navigation for drafting. Full LaTeX: `main.tex`. Experiments: `EXPERIMENT_PLAN.md`.

## Title (working)

**ConvPatch: A Controlled Study of Convolutional vs. Linear Patch Embedding in Vision Transformers**

Alternatives:
- *Patchify or Convolve? Isolating the Effect of Patch Embedding in Vision Transformers*
- *Beyond Linear Patchify: A Factorial Study of Convolutional Token Embedding in ViT*

## One-sentence pitch

We change only the ViT patch-embedding stem—holding tokens, encoder, and recipe fixed—and show which conv designs beat linear patchify, why, and at what cost.

## Section checklist

| § | Section | Status | Notes |
|---|---------|--------|-------|
| — | Abstract | Draft | Needs numbers after Phase 1 |
| 1 | Introduction | Draft | Motivation + RQ + contributions |
| 2 | Related work | Outline | Position vs Xiao/CvT/Swin/CETNet |
| 3 | Method | Draft | Variants A–E, token invariant, encoder |
| 4 | Experimental setup | Draft | Datasets, recipe, metrics, impl |
| 5 | Experiments | Draft | Phases 1–5 as subsections |
| 6 | Results | Placeholder | Tables TBD |
| 7 | Discussion | Placeholder | Mechanism + limitations |
| 8 | Conclusion | Draft | One paragraph |

## Figures to produce

1. **Teaser:** Linear vs conv stem diagram (image → tokens → fixed encoder)
2. **Design space:** Five variants side-by-side (from DESIGN.md Fig. 3)
3. **Main results:** Grouped bar chart, variant × dataset
4. **Convergence:** Val acc vs epoch, A vs B (+ shading over seeds)
5. **Low-data:** Gap vs training fraction
6. **LR sensitivity:** Line plot or heatmap
7. **Ablations:** A4 pos-emb × stem; A6 param-matched

## Tables to produce

1. **Tab. 1:** Main top-1 (variants × datasets × ViT-Ti)
2. **Tab. 2:** ViT-S row / extension
3. **Tab. 3:** Low-data (10%, 25%)
4. **Tab. 4:** Ablations A4, A6
5. **Tab. 5:** mCE robustness
6. **Tab. 6:** Params, FLOPs, throughput, memory

## Reviewer-facing strengths

- Clean causal story (one module changed)
- Token-count invariant (unlike ViT_C block removal)
- Param-matched control (A6)
- Reproducible configs + seeds
- Honest compute scope (no inflated IN-1k claims)

## Reviewer-facing risks (address proactively)

- "Small datasets only" → frame as controlled study; IN-100 as transfer check
- "Xiao et al. already did this" → constant N, broader design space, ablations
- "Only tiny models" → ViT-S included; scaling in future work
- "Cherry-picked conv variant" → report all five; pre-specify B vs A as primary

## Writing order (recommended)

1. Method §3 (mostly done in code — write from `patch_embed.py`)
2. Experimental setup §4 (from configs)
3. Experiments §5 (from EXPERIMENT_PLAN.md)
4. Introduction §1 (once claims are crisp)
5. Related work §2
6. Results §6 (after first training runs)
7. Abstract last
