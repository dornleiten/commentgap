# Legacy analyses

This directory preserves the earlier combined-paper workflow and pre-2025 data
analyses. These files are references, not stages in the 2025 Paper 1 pipeline.

- `01_*`: the former RSelenium scrape.
- `02A`–`02F`: earlier preference, gap, regression, and ranking models.
- `03`–`05`: FORUM, ranking-policy, and publication outputs from the earlier
  combined paper.
- `02E_better-gap-measures.Rmd` and `02E_gap_figs.ipynb`: the old gap-score
  implementation and figures, retained as the design reference for a new 2025
  implementation.

Most legacy files assume they are launched from the repository root and may read
the tracked top-level `model_output/` or `models/` artifacts. Moving those outputs
under `legacy/` would require updating and validating all historical paths first.
