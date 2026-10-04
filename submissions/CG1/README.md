# CG1 submission

This directory contains the preserved CG1 submission: manuscript sources, bibliography and style files, required `.bbl`, figures, tables, and the submitted PDF. Their expected SHA-256 values are recorded in [the paper asset registry](../../provenance/paper-assets.csv).

CG1 studies reader and journalist preferences in the retrospective 2025 forum collection. The analysis includes the comment-gap measure, descriptive and topic summaries, conditional-logit selection models with sensitivity analyses, XGBoost and neural rankers, and reporting products.

The numbered workflow is stages 01–09. Frozen presentation replay is documented in [`docs/reproduction.md`](../../docs/reproduction.md); it reads retained stage products and writes to `outputs/<run_id>/`.

To verify preserved bytes:

```bash
python scripts/validate_publications.py
```

To compile an isolated copy, use a new run ID and a working TeX installation:

```bash
python scripts/validate_publications.py --compile --run-id cg1-publication-check
```
