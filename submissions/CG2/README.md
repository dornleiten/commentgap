# CG2 submission

This directory contains the preserved CG2 submission: manuscript sources, bibliography and style files, required `.bbl`, figures, tables where present, and the submitted PDF. Their expected SHA-256 values are recorded in [the paper asset registry](../../provenance/paper-assets.csv).

CG2 studies ranking-algorithm effects on forum outcomes and ranking-induced topic distributions. It uses the retained FORUM analysis, policy-similarity products, and the registered topic runs and diagnostics listed in [`docs/artifacts.md`](../../docs/artifacts.md).

The numbered workflow is stages 10–16 and depends on shared preparation and frozen CG1 ranker handoffs. Frozen presentation replay is documented in [`docs/reproduction.md`](../../docs/reproduction.md); it reads retained products and writes to `outputs/<run_id>/`.

To verify preserved bytes:

```bash
python scripts/validate_publications.py
```

To compile an isolated copy, use a new run ID and a working TeX installation:

```bash
python scripts/validate_publications.py --compile --run-id cg2-publication-check
```
