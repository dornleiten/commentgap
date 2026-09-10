# Paper 2: topic agenda analysis

This workflow operationalises the topic-modelling plan only. It treats ranking
as a visibility intervention on a fixed set of comments and asks how the
presented topic agenda differs from the article and from the complete
discussion.

## Estimands

For each story, let `A` be the article topic distribution, `C` the mean topic
membership over all eligible comments, and `V(k)` the mean membership of the
comments shown by a policy through depth `k`.

- Article/discussion similarity: square-root Jensen–Shannon distance
  `D_JS(A, C)`. Natural logarithms are used, so this distance is in nats and
  ranges from 0 to `sqrt(log(2))` approx. 0.833, rather than from 0 to 1.
- Concentration: Shannon entropy `H(P)` and effective topic count `exp(H(P))`.
- Secondary similarity: ordinary cosine similarity on the original topic proportions. The square-root-space cosine is retained only as an explicitly named Bhattacharyya diagnostic because it is algebraically tied to Hellinger distance.
- Ranking alignment: `D_JS(A, C) - D_JS(A, V(k))`; positive means closer to
  the article.
- Ranking concentration: `H(V(k)) - H(C)`; positive means more distributed.
- Overshoot: Hellinger projection `alpha` of `V(k) - C` onto `A - C`.
  Values below zero move away, values from zero to one partially converge,
  values around one match, and values above one overshoot.
- Off-axis reconstruction: the Hellinger residual from that projection.

Topic-specific gap ratios are secondary and are suppressed when the baseline
article–discussion gap is small. All metrics are calculated in one common topic
space; separate article and comment models must not be compared directly.

## Workflow

1. Run `14_topic_model_fit.ipynb` to load the collected corpus, fit a shared BERTopic model on Paper 2 passages plus an equally sized eligible comment sample, with automatic topic reduction and the precomputed BGE-M3 embeddings, then transform article passages and discussion documents into that fixed topic space, inspect topic terms, and write document-topic memberships.
2. Run `15_topic_agenda_analysis.ipynb` to aggregate article and
   complete-discussion distributions, construct deterministic policy/random
   rankings from the frozen Paper 2 score table, calculate depth trajectories,
   and write paired contrasts and plots.

The model-fitting stage can also be run headlessly:

```bash
.venv/bin/python scripts/run_topic_model_fit.py
```

Use `--all-stories` to override the default development-only fit. The runner fits on articles only by default; use `--include-comments-in-fit` only for a deliberately smaller corpus or a machine with sufficient memory.

The notebooks are deliberately read-only by default. Set `RUN_SEARCH = True` and/or `RUN_FINAL = True`
in notebook 14 only after reviewing the model configuration. The backend is
[`commentgap_analysis/topic_modeling.py`](../../commentgap_analysis/topic_modeling.py).

## Reproducibility and interpretation

Notebook 14 fits on Paper 2 stories (`--fit-role paper2_test`) and passes its exact
story-balanced, size-matched comment sample to the runner. This is corpus-level
measurement; the runner retains development fitting as its CLI default for legacy
workflows. Final transformations use one consistent assignment rule for all Paper 2
documents. German/English vectorizer stopwords and c-TF-IDF frequent-word reduction
improve topic terms. Embedding-based outlier reduction uses a configurable cosine
threshold (initially 0.5), retaining raw assignments and reassignment flags. Review
raw/reduced coverage and semantic quality before interpreting ranking contrasts. Model size,
preprocessing, random seed, topic terms, the actual BERTopic submodel settings,
and the model artifact are saved under
`model_output/selection_2025/paper2_topic_modeling/`. The model manifest records
these settings, while `topic_model_run_metadata.json` records the split and
analysis-comment input hashes, embedding/model identity, package versions,
Python/platform, code revision, and fit scope. Notebook 15 writes analogous
run metadata and cache manifests; cached policy and oracle outputs are reused
only when their input and configuration signatures still match.

These outcomes describe the visible representation of already-produced
discourse. They do not identify effects on what users later write, believe, or
do, and topic labels should be interpreted as model-derived indicators rather
than perfectly observed issues or viewpoints.
