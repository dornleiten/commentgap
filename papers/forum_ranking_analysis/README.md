# FORUM and ranking-algorithm effects

This analysis is implemented as a separate, manifest-driven pipeline. It consumes
frozen Paper 1 artifacts but never uses held-out outcomes to select predictive
rankers. The legacy notebooks and papers remain design references only; their
outputs are not read by the new pipeline.

## Questions and estimands

The analysis asks:

1. How much does comment ordering change the attributes presented near the top
   of a discussion?
2. How do reply containment and pinned-comment structure alter those effects?
3. Which policy bundles offer defensible trade-offs across similarity,
   diversity, novelty, and toxicity?
4. When FORUM and nDCG evaluate the same complete permutation, how closely do
   their policy conclusions agree?

FORUM is the primary presentation-effect estimand. It compares a policy's
cumulative outcome trajectory with the expected trajectory under random
ordering, scaled separately toward the best and worst attainable trajectories.
Scores therefore lie in [-1, 1]; zero is the random-exposure anchor.

Because positive and negative deviations use different attainable denominators,
the mean FORUM score over realized random permutations need not equal zero,
especially for skewed outcomes at shallow depth. The explicit random policy is
therefore the empirical grounding condition as well as the regression reference.

## Frozen analysis design

The primary universe is the exact Paper 1 all-comment held-out partition stored
as paper2_test, restricted to discussions with at least 11 comments. The stored
split label predates the paper separation but is intentionally retained to
preserve the frozen test boundary. Discussions with at least 100 comments form
the prespecified large-thread sensitivity.

Predictive rankers are selected using development cross-validation only, one
winner in each family-by-feature-set cell:

- XGBoost with metadata;
- XGBoost with metadata plus BGE text;
- neural ranker with metadata;
- neural ranker with metadata plus BGE text.

Each contributes audience and editor orderings. The stacked regression audience
and editor scores are also carried forward. Winner identities and held-out score
file hashes are frozen before held-out scores are opened.

The 15 ordering conditions are:

- relative votes, upvotes, chronological, and reverse chronological;
- regression audience and editor;
- XGBoost metadata audience and editor;
- XGBoost metadata-plus-text audience and editor;
- neural metadata audience and editor;
- neural metadata-plus-text audience and editor;
- seeded random ordering as an explicit grounding control.

Crossing these with loose, tree-preserving, and root-only/hidden-reply
presentation and with unpinned/pinned states yields 90 evaluated bundles: 84
substantive bundles and six random controls. Random bundles are not treated as
deployable policies and are excluded from substantive rankings. The `random` | `loose` | `unpinned` cell is the treatment-coded regression reference.

Tied substantive scores are averaged over 10 deterministic tie draws. Random
controls use 100 deterministic permutations by default. These are separate
sources of uncertainty.

## Outcomes

Primary comment attributes are:

- expected aggregate AQuA deliberative quality;
- similarity to the article's three closest passages;
- maximum toxicity probability across text chunks;
- participant incumbency, measured as logged author comments in the prior 30 days;
- prior audience reception, measured as logged prior upvote reception minus
  logged prior downvote reception;
- comment novelty, defined as mean cosine distance to the five nearest other
  comments in the completed discussion;
- positive and negative sentiment probabilities separately;
- raw CTTR; and
- raw SMOG-DE.

Five-neighbour novelty is deliberately not the strict-prior novelty used in Paper 1.
It is invariant to the counterfactual policy order. Exact blockwise search is
used through 5,000 comments; larger discussions use deterministic HNSW with an
exact validation sample and a minimum 0.95 recall requirement.

The at-least-100-comment sample and top-10 versus full trajectories remain
prespecified sample and depth analyses.

The canonical run scores only these selected outcomes; it does not add alternative
toxicity or novelty definitions, semantic atypicality, or individual AQuA dimensions.


## Interface semantics
Loose mode reorders every comment independently. Tree mode ranks roots and then
renders each complete reply subtree in frozen preorder. Hidden mode ranks roots
only; FORUM linearly interpolates the unobserved remainder to the discussion
total, matching the legacy estimand without pretending to have a complete
visible permutation.

If a deleted ancestor is absent from the Paper 1 choice set, its first surviving
children become roots of the induced visible forest. This retains every eligible
reply without assigning a ranking score to an unobserved deleted comment.

Pinning respects the selected interface:

- loose mode moves every sticky comment to the front in observed display order;
- tree and hidden modes move sticky visible roots only;
- a sticky reply remains inside its subtree in tree/hidden modes.

Direct continuous-relevance nDCG is calculated for every complete policy order.
Loose and tree policies already produce complete permutations; hidden-reply
policies are completed in frozen preorder solely for this nDCG diagnostic, while
FORUM retains its separate hidden-reply interpolation estimand. Metric agreement
is reported within each reply-mode × pin-state group and for all 84 substantive
policy bundles overall.

## Inference and reporting products

The discussion is the paired resampling unit. The default analysis uses 2,000
seeded bootstrap draws and produces:

- policy-cell FORUM means and 95% percentile intervals;
- ordering-versus-random, reply-versus-loose, and pinned-versus-unpinned
  average marginal contrasts;
- a two-way ordering/reply/pinning decomposition with
  random | loose | unpinned as reference, plus cell residuals that expose
  omitted three-way structure;
- FORUM/nDCG rank agreement for actual loose, unpinned permutations;
- within-discussion Spearman alignment between every predictive score and each
  primary outcome.

Sentiment, reading difficulty, AQuA, and author attributes are reported but
excluded from the policy-ranking summaries. Random controls are also excluded
from deployable policy rankings.

## Runbook

Install the repository in the project environment so the entry point is current:

~~~bash
.venv/bin/python -m pip install -e .
~~~

Run stages independently so each manifest can be inspected before the held-out
pipeline advances:

~~~bash
commentgap-forum-analysis freeze

commentgap-forum-analysis build \
  --embedding-store PATH_TO_COMPLETE_BGE_STORE

commentgap-forum-analysis score

# Optional: score independent discussions in parallel. Notebook 10 passes the
# extended FORUM outcome set; the CLI defaults to the canonical FORUM outcome set.
commentgap-forum-analysis score --workers 8

commentgap-forum-analysis infer

commentgap-forum-analysis report
~~~

Or run all five stages in sequence:

~~~bash
commentgap-forum-analysis all \
  --embedding-store PATH_TO_COMPLETE_BGE_STORE
~~~

The freeze stage refuses to run while the factorial ranker process is active.
Do not use --allow-active-factorial for a production freeze.

## Outputs and resumability

All new artifacts live below
model_output/selection_2025/forum_ranking_analysis/:

- ranker_handoff/: development-only winner tables and immutable file hashes;
- static_novelty/: one static-novelty checkpoint per discussion;
- analysis_comments.parquet: the joined, validated FORUM analysis comment table;
- policy_scores/stories/: one resumable 90-bundle score checkpoint per
  discussion;
- policy_scores/policy_scores.parquet: the consolidated score panel;
- inference/: CSV estimates, story-level mechanism correlations, and a final
  input/output manifest.
- reporting/: publication-ready PDF/PNG figures, the primary LaTeX table, and
  a hash manifest.

Changing a policy-scoring signature invalidates only the affected story
checkpoints. Every consolidated input and output is hashed. A production run
should archive the handoff, analysis, policy-score, and inference manifests with
the paper materials.

## Implementation

The analysis engine is
[commentgap_analysis/forum_scores.py](../../commentgap_analysis/forum_scores.py),
the command interface is
[commentgap_analysis/forum_analysis_cli.py](../../commentgap_analysis/forum_analysis_cli.py),
the publication-output layer is
[commentgap_analysis/ranking_algorithm_effects.py](../../commentgap_analysis/ranking_algorithm_effects.py),
and focused invariance/semantics tests are in
[tests/test_forum_scores.py](../../tests/test_forum_scores.py).
