# Paper 1: reader and journalist preferences

This is the authoritative order for the 2025 preference paper. The primary
candidate set is `all`. The `root` candidate set is retained as an appendix
sensitivity and runs only when explicitly requested.

## Scope contract

- Primary paper scope: `all` comments.
- Appendix scope: `root` comments.
- Selection labels: curator/editor picks and audience top-k selections, with ten
  deterministic audience tie draws.
- Model selection data: the five development folds only.
- Final evaluation data: the held-out 50% article partition. Its persisted role
  is still named `paper2_test` for backward compatibility; in Paper 1 prose it
  should be called the held-out test partition.
- Scope selection for R: `COMMENTGAP_MODEL_SCOPES=all` by default, or
  `COMMENTGAP_MODEL_SCOPES=all,root` for the appendix.
- Scope selection for the factorial CLI: pass `--scope all`; after the current
  run finishes, the CLI default should be changed from both scopes to `all`.

## Pipeline

| Stage | Entry point | Status and contract |
| --- | --- | --- |
| 1. 2025 scrape | [`01_scrape_2025.ipynb`](../../01_scrape_2025.ipynb) and `commentgap-scrape` | Implemented. Collection completeness, privacy metadata, hash-key continuity, and QA must pass before inference. December 2024 is a feature lookback, not part of the 2025 outcome sample. |
| 2. Build model features | [`02_build_model_features.ipynb`](../../02_build_model_features.ipynb) | Implemented. Produces resumable embeddings, similarities, AQuA, scalar features, and matched `all`/`root` choice sets. |
| 3. Shared preprocessing | [`03_shared_model_preprocessing.ipynb`](../../03_shared_model_preprocessing.ipynb) | Implemented, but bootstraps from an ignored legacy split at `xgboost_paper2/master_article_split.parquet`. A clean clone cannot reproduce the exact assignment without that external artifact. |
| 4. Feature diagnostics | [`04_feature_distribution_diagnostics.ipynb`](../../04_feature_distribution_diagnostics.ipynb) | Implemented. It diagnoses `model_data/`, the transformed contract actually consumed downstream, while retaining the source feature-build watermark check. |
| 5. Descriptives/topic | [`05_descriptive_statistics_topics.ipynb`](../../05_descriptive_statistics_topics.ipynb) and [`scripts/run_paper1_descriptives.py`](../../scripts/run_paper1_descriptives.py) | Implemented and fixture-tested. Produces manifested sample-flow, month, uncollapsed `section_1` topic, hierarchy-audit, discussion, feature, and figure artifacts. Audience selections are averaged over ten tie draws. |
| 6. Comment-gap score | [`06_comment_gap_score.ipynb`](../../06_comment_gap_score.ipynb) and [`scripts/run_comment_gap.py`](../../scripts/run_comment_gap.py) | Implemented and fixture-tested in Python. Preserves the corrected legacy midrank definition on `all` by default, adds ten-draw overlap diagnostics, and writes article/topic summaries and figures. The old Rmd remains a methodological reference only. |
| 7. Stacked selection models | [`07_stacked_selection_models.Rmd`](../../07_stacked_selection_models.Rmd) | Implemented. Defaults to `all`; `root` is opt-in. Uses development articles for fitting and the held-out partition only for frozen evaluation. |
| 7A. Robustness | [`07A1_collinearity_sensitivity.Rmd`](../../07A1_collinearity_sensitivity.Rmd) and [`07A2_efron_exact_sensitivity.Rmd`](../../07A2_efron_exact_sensitivity.Rmd) | Implemented as two analyses and one combined render stage. Both default to `all`; `root` is opt-in. They use development data only for specification checks. |
| 8. XGBoost/neural variants | [`08_xgboost_neural_model_variants.ipynb`](../../08_xgboost_neural_model_variants.ipynb), protected execution record [`06C5_ranker_factorial.ipynb`](../../06C5_ranker_factorial.ipynb), [`scripts/run_ranker_factorial.py`](../../scripts/run_ranker_factorial.py), and [`scripts/freeze_factorial_winners.py`](../../scripts/freeze_factorial_winners.py) | The 68-model factorial is the only Paper 1 ML workflow. Training is in progress. The fixture-tested finalizer refuses to run while the launcher is active, requires exact five-fold coverage for every planned variant/scope, and selects XGBoost/neural winners from development CV only. |
| 9. Tables and plots | [`09_model_tables_plots.ipynb`](../../09_model_tables_plots.ipynb) and [`scripts/run_paper1_reporting.py`](../../scripts/run_paper1_reporting.py) | Implemented and fixture-tested. Validates frozen CV/plan/ranking hashes before opening held-out artifacts; combines stage 7 with the XGBoost and neural winners; writes CSV/LaTeX tables, paired article-bootstrap comparisons, tie sensitivity, and PNG/PDF figures. Full-data execution waits for stage 8 completion. |

### Comment-gap interpretation

The normalized rank gap is the primary measure. The legacy-style Jaccard
overlap, and its complement reported as the Jaccard gap, are secondary
descriptive comparisons. Jaccard discards the rank within each selected set,
depends on the number of curator picks and audience tie policy, and may reflect
exposure feedback because pinned comments can receive more votes. It should
therefore not be presented as a definitive or causal measure of preference
disagreement, and comparisons across studies require matching the sample and
top-k construction.

After the active factorial run finishes, build the all-comment descriptive and
gap artifacts, then render the regression and robustness stages with:

```bash
.venv/bin/python scripts/run_paper1_descriptives.py
.venv/bin/python scripts/run_comment_gap.py
Rscript scripts/render_rmd.R 07_stacked_selection_models.Rmd
Rscript scripts/render_rmd.R \
  07A1_collinearity_sensitivity.Rmd \
  07A2_efron_exact_sensitivity.Rmd
.venv/bin/python scripts/freeze_factorial_winners.py
.venv/bin/python scripts/run_paper1_reporting.py
```

Add the root appendix with:

```bash
COMMENTGAP_MODEL_SCOPES=all,root \
  Rscript scripts/render_rmd.R 07_stacked_selection_models.Rmd
.venv/bin/python scripts/run_paper1_descriptives.py --scope all --scope root
.venv/bin/python scripts/run_comment_gap.py --scope all --scope root
.venv/bin/python scripts/freeze_factorial_winners.py --scope all --scope root
.venv/bin/python scripts/run_paper1_reporting.py --scope all --scope root
```

The same environment variable works for both robustness documents.

## Model-selection rule

Stage 8 tunes hyperparameters within every variant, then
`scripts/freeze_factorial_winners.py` freezes the cross-variant winners. It
requires all five folds and ranks within each scope and family by mean
development-CV macro nDCG@k, lower fold SD, higher minimum fold score, then
stable variant ID. The Paper 1 primary winners are the `all` XGBoost winner and
the `all` neural winner. Their held-out metrics are then reported once; `root`
winners belong in the appendix.

The freezer does not read `sealed_test_results.csv`. Because several earlier
workflows and all factorial variants already calculate this partition, describe
it conservatively as held-out rather than claiming that it remained operationally
unseen throughout model development.

## Known inconsistencies and failure points

1. `paper2_test`, `sealed_paper2_test`, `xgboost_paper2`, and
   `reporting_paper2` are legacy schema/path names now attached to Paper 1 model
   evaluation. Rename them only through a versioned migration after the current
   factorial run; changing them piecemeal will invalidate caches and readers.
2. The frozen split needed by stage 3 is under ignored `model_output/`. Stage 3
   does not create a replacement: by default it reads
   `model_output/selection_2025/xgboost_paper2/master_article_split.parquet`,
   validates it, and copies the assignment into `model_data/`. Since the whole
   parent tree is gitignored, a clean clone has the code and seed but not the
   exact story-to-partition/fold assignment or its source hash. Re-running later
   could therefore yield a different split if the eligible story set or split
   algorithm changes. Fix this after the factorial by either publishing the
   privacy-safe `story_id` assignment plus SHA-256, or by tracking a deterministic
   generator whose output is required to match the recorded hash.
3. The original AQuA promotion audit note is absent because `/notes/*` is
   ignored. The feature notebook now records that provenance gap rather than
   linking to a nonexistent tracked document.
4. Stage 5 joins article metadata because `section_1`, `section_2`, and
   `section_3` do not survive into the model choice sets. `section_1` is now the
   uncollapsed primary topic; sections 2–3 are retained in the audit table.
5. The legacy gap score was root-only. The 2025 implementation resolves this by
   making `all` primary and retaining `root` as an opt-in appendix sensitivity.
6. `06A2_shared_model_preprocessing.executed.ipynb` and tracked `.orig` source/
   test files are development artifacts, not pipeline stages. Move or remove
   them after checking whether they contain the only record of an important run.

## Active CUDA factorial run: protected artifacts

The run was launched from the repository root as:

```text
.venv/bin/python -u scripts/run_ranker_factorial.py --include 'xgb__*' --xgb-device cuda
```

The earlier CPU attempt received `KeyboardInterrupt`; this CUDA command resumed
the checkpoints and was live when the reorganization was performed. It still
uses the runner's existing default scope order, `root` then `all`. Until the CUDA
worker and screen launcher have completed, do not move, rename, edit, or delete:

- `scripts/run_ranker_factorial.py`;
- `06C5_ranker_factorial.ipynb`;
- `commentgap_analysis/factorial_rankers.py`, `ranking.py`,
  `neural_ranking.py`, or `preprocessing.py`;
- `data/scrape_2025/`;
- `model_output/selection_2025/model_data/`;
- `model_output/selection_2025/embeddings/`;
- `model_output/selection_2025/factorial_rankers/`, including its shared BGE
  memory-map cache and experiment-wide summary CSVs; or
- `logs/06C5_ranker_factorial_xgb_cuda.log` and the active environment.

After completion, change the factorial CLI default to `all`, migrate the stale
Paper 2 labels as one atomic compatibility change, and move the protected `06C5`
execution record out of the numbered pipeline. The canonical Stage 8 finalizer
is already `08_xgboost_neural_model_variants.ipynb`.

The exact deferred actions, including execution of the development-CV freezer,
Stage 9 report, and compatibility-label migration, are recorded in
[`POST_FACTORIAL_MIGRATION.md`](POST_FACTORIAL_MIGRATION.md).
