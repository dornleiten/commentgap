# Script entry points

The files at this level support the current public Replay, private Recompute,
and Fresh workflows:

- `canonical_artifacts.py` verifies the public bundle and manages reviewed
  private promotions. `build_public_replay.py` prepares the public bundle from
  selected private products. `validate_publications.py` checks paper assets.
- `render_rmd.R`, `paths.R`, and `path_contract.py` support the R Markdown
  documents. `frozen_rmd_load.R` loads their private Recompute inputs.
- The `build_*.py`, `run_*.py`, `freeze_factorial_winners.py`, and
  `compute_cg1_baselines.py` entry points support Fresh production and
  publication reporting. Stage 08 invokes `run_ranker_factorial.py`; stage 14
  invokes `run_topic_model_fit.py`.

`historical/` preserves source used in one-time reconstructions. Those files
are evidence for how accepted artifacts were derived and are not part of the
current notebook execution path. Some recorded source hashes depend on their
original bytes, so the files were moved without editing them. Scripts that
derive the repository root from their own location must be restored to their
original `scripts/` path before rerunning against private data.

The local, Git-ignored `private/` directory contains the completed migration,
bundle restoration, and older paper replay commands. It is absent from a
public checkout, which uses `artifacts/canonical/` for Replay.
