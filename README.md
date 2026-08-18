# The News Comment Gap and Algorithmic Agenda Setting in Online Forums
Project repository

* 01_Code: used for scraping derstandard.at with RSelenium
* 02_Code: used for data cleaning, and analysis of comment preferences for readers/journalists
* 03_Code: used for analysing ranking algorithms
* 04_Code: used for beta regression model for FORUM score
* 05_Code: used for creating figures and tables for the article

### The News Comment Gap and Algorithmic Agenda Setting in Online Forums
#### Flora Böwing, Patrick Gildersleve

_The disparity between news stories valued by journalists and those preferred by readers, known as the "News Gap", is well-documented. However, the difference in expectations regarding news related user-generated content is less studied. Comment sections, hosted by news websites, are popular venues for reader engagement, yet still subject to editorial decisions. It is thus important to understand journalist vs reader comment preferences and how these are served by various comment ranking algorithms that represent discussions differently. We analyse 1.2 million comments from Austrian newspaper Der Standard to understand the "News Comment Gap" and the effects of different ranking algorithms. We find that journalists prefer positive, timely, complex, direct responses, while readers favour comments similar to article content from elite authors. We introduce the versatile Feature-Oriented Ranking Utility Metric (FORUM) to assess the impact of different ranking algorithms and find dramatic differences in how they prioritise the display of comments by sentiment, topical relevance, lexical diversity, and readability. Journalists can exert substantial influence over the discourse through both curatorial and algorithmic means. Understanding these choices' implications is vital in fostering engaging and civil discussions while aligning with journalistic objectives, especially given the increasing legal scrutiny and societal importance of online discourse._

Read the [preprint on arXiv](https://arxiv.org/abs/2408.07052)

You can try out our FORUM score for evaluating ranking algorithm performance using this repo: [PyFORUM](https://github.com/pgilders/pyforum)

## Retrospective 2025 forum collection

The repository now includes `commentgap_scraper`, a resumable Python collector for
articles published in 2025 and the comments currently attached to them. It uses the
site's monthly sitemaps and public forum GraphQL endpoint; it does not automate a
browser, log in, post, or rate content.

The resulting dataset is a **collection-time snapshot**. In particular, vote totals,
deleted status, pinned status, and profile attributes reflect their state when the
crawl runs, not necessarily their state at the end of 2025.

### Setup

Use Python 3.11 or newer and install the collection dependencies:

```bash
python -m pip install -r requirements-scraper.txt
python -m pip install -e .
```

Set an identifying user agent, a monitored contact address, and a stable secret used
to pseudonymize authors. Keep the same secret for every resumed run, and do not add it
to the repository or distribute it with the data.

```bash
export COMMENTGAP_USER_AGENT="CommentGap academic research crawler"
export COMMENTGAP_CONTACT="researcher@example.org"
export COMMENTGAP_HASH_KEY="replace-with-a-long-random-secret"
```

Review [Der Standard's robots policy](https://www.derstandard.at/robots.txt) and
[current terms](https://about.derstandard.at/agb/) before collecting or distributing
data. The collector enforces a minimum one-second interval between request starts.

### Commands

```bash
# Discover and checkpoint every article in the 12 monthly 2025 sitemaps.
commentgap-scrape discover --year 2025

# Recommended small live smoke test before a full run.
commentgap-scrape crawl --year 2025 --limit 100 --selection stratified-pilot \
  --selection-seed 2025 --pilot-candidate-pool 500
commentgap-scrape validate --year 2025 --allow-incomplete

# Resume pending/in-progress work; add --retry-failed to revisit failures.
commentgap-scrape crawl --year 2025
commentgap-scrape crawl --year 2025 --retry-failed

# Write normalized compatibility tables for the existing R analysis.
commentgap-scrape export-legacy --year 2025
```

To add comments attached to articles published in December 2024 to the same dataset,
use the same output directory and, critically, the exact original 2025 hash key:

```bash
commentgap-scrape discover --year 2024 --month 12 --output data/scrape_2025

# The first run verifies the key against existing public 2025 forum records,
# then registers a non-secret fingerprint; the key itself is never stored.
commentgap-scrape crawl --year 2024 --month 12 --output data/scrape_2025

commentgap-scrape validate --year 2024 --month 12 --output data/scrape_2025
```

The first v0.4.0 crawl tries to reproduce existing hashes using current public forum
records and stops if the supplied key demonstrably differs. If changing or inaccessible
forum records make that check inconclusive, it asks for the one-time
`--confirm-existing-hash-key` flag; use that override only after independently checking
that the exact original 2025 key is set. Older outputs deliberately retain no raw
author identifiers, so an entirely offline retrospective check is impossible. After
registration, the crawler checks every supplied key against `privacy_metadata.json`
before writing collection output and refuses a mismatch. The same HMAC scheme and key
then produce join-compatible `author_hash` values in the 2024 and 2025 partitions.

The month refers to the article publication month, not the comment creation month.
Comments added to those December 2024 articles during 2025 or later are also present
in this retrospective snapshot. For a strictly pre-2025 author-activity feature,
filter comment `created_at` to before `2025-01-01T00:00:00` in Europe/Vienna.

Use repeated `--story-id ID` arguments to target known fixtures or unusual forums.
Outputs default to `data/scrape_2025/`, which is ignored by Git. The main tables are
partitioned Parquet datasets named `articles`, `forums`, `forum_pages`, and `comments`; the SQLite
manifest and page-level staging files make interrupted crawls resumable. Validation
writes both JSON and Parquet summaries under `qa_summary`. Month-scoped validation is
stored below `qa_summary/year=YYYY/month=MM`; collection metadata is likewise scoped
by year/month so adding 2024 does not overwrite the completed 2025 metadata.

`stratified-pilot` first inspects the forum counters of a bounded, seeded candidate
pool, then balances the selected stories across months and the bands no forum, no
postings, 1–49, 50–249, 250–999, and 1,000+ postings. At the required one-second
interval, the default 500-candidate preflight takes at least about eight minutes. Its
selection and achieved bands are recorded in `pilot_selection.json`.

Progress is written to the normal log stream so it remains readable in an interactive
terminal, a redirected log file, or an unattended job. Crawl updates include overall
percentage, elapsed time, ETA, throughput, and status counts. Large forums also report
page and posting progress every ten GraphQL pages; adjust this with
`--progress-every-pages N`.

Comments retain the API's separate `title` and `text` fields. `effective_text` is the
analysis-ready combination: when both are present it is `title`, one newline, then
`text`; when only one is present it uses that field. Deleted tombstones may have no
effective text. The legacy export provides the same newline-separated value as
`heading_and_text.comment`.

`totalPostingCount` is treated as an advisory forum counter. If a complete cursor
walk does not reconcile, the scraper refreshes the lightweight forum counter but
does not repeat the article or comment-page requests. A remaining discrepancy becomes
the explicit terminal state `completed_with_count_discrepancy`, rather than a failure.
The forum row records reported, unique, published, and deleted counts; signed and
absolute differences; and percentage discrepancy. `count_discrepancy_reproduced` is null for new records
because no second crawl was performed; older audited records retain their historical
true/false value.
Validation accepts this state while continuing to reject broken pagination, trees,
or comment-to-forum references.
Each new crawl also writes per-page diagnostics without retaining response payloads:
hashed cursors, root-edge and flattened-record counts, page completion, and cursor
progression. Aggregate diagnostics are repeated in the forum row for convenient QA.

To migrate a pilot produced by the earlier strict-count version, use the original
hash key and rerun only its failed stories:

```bash
commentgap-scrape migrate-existing --year 2025
commentgap-scrape crawl --year 2025 --only-failed
commentgap-scrape validate --year 2025 --allow-incomplete
```

`--only-failed` includes both failed and interrupted `in_progress` stories while
excluding untouched pending stories. The retry removes legacy orphan comment files
for count-mismatch failures and replaces them only after a complete, internally valid
cursor walk reaches a terminal state. Keeping the original `COMMENTGAP_HASH_KEY`
preserves author pseudonyms across the pilot.
The offline migration adds newline-separated `effective_text`, clears incorrectly
inherited sticky flags only in historical files without page evidence, and upgrades
old forum schemas. Current validation reconciles retained sticky-record counts against
the sticky API count because an individually sticky posting can legitimately be nested
in its normal thread. The migration does not invent diagnostics that were not retained
originally.

The compatibility export intentionally keeps article text in a separate article
table rather than duplicating it for every comment. `user_names` contains the stable
pseudonym, and timestamp columns use ISO 8601 UTC values.

Unauthenticated article responses do not always expose a publication timestamp. In
that case, `published_at` uses the sitemap's `lastmod` value and
`published_at_source` is `sitemap_lastmod_fallback`. Because a sitemap timestamp can
also reflect an article update, analyses that depend on exact publication time should
filter or sensitivity-check this field. The original sitemap value is retained in
`sitemap_lastmod` for auditability.

## Four-model 2025 preference analysis

The `06` workflow compares curator and audience top-k selections on two
candidate sets. It fits a root-only and an all-comment stacked conditional-logit
model, plus matching article-grouped XGBoost rankers:

1. `06A_build_model_features.ipynb` creates resumable scalar features and
   matched choice sets from the normalized Parquet collection.
2. `06B_stacked_selection_models.Rmd` fits both conditional-logit regressions
   with article-by-selector strata and article-clustered standard errors.
3. `06C_xgboost_rankers.ipynb` performs separate tuning, five-fold
   article-grouped OOF evaluation, and final full-data fits for both scopes.
4. `06D_model_tables_plots.ipynb` exports CSV/LaTeX tables and SVG/PDF figures.

Install the tested Python environment and restore the R environment before a
production run:

```bash
python -m pip install -r requirements-analysis.txt
python -m pip install -e .
Rscript -e 'if (!requireNamespace("renv", quietly=TRUE)) install.packages("renv"); renv::restore()'
```

The default notebook settings are inference-safe: feature extraction refuses
an incomplete crawl or pilot NLP. A small pipeline check on the current partial
collection must be explicitly watermarked, for example:

```bash
export COMMENTGAP_INFERENCE_MODE=0
export COMMENTGAP_ALLOW_INCOMPLETE=1
export COMMENTGAP_NLP_MODE=pilot
export COMMENTGAP_MAX_STORIES=40
export COMMENTGAP_EXCLUDE_JANUARY_WITHOUT_LOOKBACK=0
```

Never use those pilot outputs in a paper. For inference, unset all five pilot
variables, complete and validate the crawl, and provide
`COMMENTGAP_LOOKBACK_ROOT` pointing to a consistently pseudonymized December
2024 collection. Without that lookback, January 2025 is excluded from the
author-history models. `COMMENTGAP_DEVICE=auto` uses CUDA for production NLP on
a compatible Linux installation and otherwise uses MPS/CPU as supported;
XGBoost uses CUDA when its installed build supports it and CPU histograms on
macOS.

Production text classification uses immutable checkpoint revisions of
`cardiffnlp/twitter-xlm-roberta-base-sentiment` and
`textdetox/xlmr-large-toxicity-classifier-v2`. The feature outputs retain
positive, negative, and neutral sentiment probabilities. Toxicity is recorded
as both the maximum chunk probability (`toxicity_probability`, used by the
models) and a token-weighted mean (`toxicity_mean_probability`, descriptive
only), so localized abuse is not diluted in long comments. The selected model
IDs, revisions, and aggregation rules are written to `feature_manifest.json`.
They can be overridden from `commentgap-features` with
`--sentiment-model-id`, `--sentiment-revision`, `--toxicity-model-id`, and
`--toxicity-revision`; custom revisions are resolved to immutable Hub commits
before inference.

### Isolated AQuA deliberative-quality store

AQuA is deliberately not installed in the main Transformers 5 environment.
The `commentgap-aqua` stage exports keyed Parquet inputs, invokes a separate
Python 3.10 runtime, validates its outputs, and writes resumable per-story
checkpoints. The committed schema freezes upstream AQuA commit
`637914dcd62491766ff478dc21632813780d005d`, multilingual BERT revision
`3f076fdb1ab68d5b2880cb87a0886f315b8146f8`, the published component order and
weights, and SHA-256 hashes for all 80 adapter files.

Create the environment separately. On macOS/CPU:

```bash
python3.10 -m venv .venv-aqua
.venv-aqua/bin/python -m pip install -r requirements-aqua-legacy.txt
```

On the Linux A5000 host, use the CUDA 11.3 lock instead:

```bash
python3.10 -m venv .venv-aqua
.venv-aqua/bin/python -m pip install -r requirements-aqua-cuda113.txt
```

Place the exact upstream checkout at
`.cache/aqua-upstream-637914d/`; the runtime expects its adapters under
`trained adapters/` and verifies every file before loading a model. A pilot
before upstream parity has been frozen must be explicitly watermarked:

```bash
commentgap-aqua \
  --device cuda \
  --max-stories 20 \
  --allow-unverified-parity
```

The default parallel composition automatically retries a story sequentially
if the accelerator reports an out-of-memory error. Use
`--execution-mode sequential` to select the lower-memory path from the start.
Production builds fail closed until `commentgap-aqua-parity` verifies the
upstream hard labels and published score, repeated CPU and sequential logits,
and updates `aqua_runtime/artifacts.json` with the fixture hash.

Each complete store retains all 20 hard labels, logits, raw four-class softmax
values, raw expected ordinal values, published hard composite score, raw
expected composite score, token count, and truncation status. The probabilities
are explicitly marked `uncalibrated`. To merge the compact hard/expected
features into the existing choice sets without loading the legacy model, pass:

```bash
commentgap-features --aqua-store model_output/selection_2025/aqua
```

The merge requires exact candidate coverage, matching text hashes and build
signatures, a completed validation report, and a production watermark. AQuA
features remain descriptive/sensitivity features and are not automatically
added to the confirmatory root/all model specifications.

Outputs are written below `model_output/selection_2025/`. Only
`oof_scores_wide.parquet` is eligible for predictive-performance claims or the
Paper 2 ranking-policy evaluation. Predictions from
`final_deployable_model.json` are full-data scores and must not be presented as
held-out results.

### Reusable embedding store

Create normalized comment and article-passage vectors independently of the
feature notebook with the installed command or its repository-local script.
By default the builder discovers every year present in both the `articles` and
`comments` datasets, currently December 2024 and all of 2025:

```bash
# Mac mini (Apple GPU)
commentgap-embed --device mps --batch-size 128 --storage-dtype float16

# Linux workstation (NVIDIA GPU)
python scripts/build_embeddings.py --device cuda --batch-size 128
```

The default checkpoint is `BAAI/bge-m3`. Any model supported by Sentence
Transformers can be selected explicitly; a model swap creates a separate,
model-versioned output namespace and cannot overwrite or mix with BGE-M3:

```bash
commentgap-embed \
  --model-id ORGANIZATION/MODEL \
  --revision COMMIT_OR_TAG \
  --device cuda
```

Pin `--revision` for paper results. Some checkpoints require a particular prompt
or asymmetric query/document treatment; consult that model's documentation before
assuming its vectors are directly comparable. The manifest records the requested
and resolved revision, maximum length, prompt, dimensions, source-data fingerprint,
packages, and device.

The default `BAAI/bge-m3` model is pinned to immutable commit
`5617a9f61b028005a4858fdac845db406aefb181`. When another model is selected and
`--revision` is omitted, the builder queries the Hugging Face Hub for that model's
current default-branch commit and then loads tokenizer and weights using the resolved
40-character SHA. Branch names and tags passed through `--revision` are likewise
resolved before loading; an already exact commit is used directly. Failure to
resolve an immutable commit stops the run rather than emitting `main_unresolved`.

Outputs live under
`model_output/selection_2025/embeddings/model=.../build=.../`, partitioned into
`comments` and `article_passages`. Every Parquet row has a stable comment or passage
key, a source-text SHA-256 checksum, and a normalized fixed-size vector in the
configured storage dtype (`float32` by default or `float16` when requested).
Raw comment text is not duplicated into this derived store. One Parquet checkpoint
is written per story; restarting the same model/data build skips finished stories.
Use `--allow-incomplete` or `--max-stories` only for explicitly watermarked pilot
stores.

An explicit `--device cuda` or `--device mps` request is validated before the
corpus run. The command prints both requested and resolved devices and stops with
an actionable error rather than silently falling back to CPU. Progress output is
printed every 25 articles by default, including percentage, newly embedded comment
and passage counts, elapsed time, throughput, and ETA. Adjust the interval with
`--progress-every-stories N`.

Embeddings are computed as normalized float32 vectors regardless of storage type.
`--storage-dtype float16` only casts the checkpointed vectors, roughly halving disk
space and later read bandwidth; it does not make transformer inference use float16.

Before committing GPU time, run the exact tokenizer-length audit on any CPU-only
machine. This loads the tokenizer but not BGE-M3 model weights and does not use a GPU:

```bash
python scripts/build_embeddings.py --diagnostics-only --max-length 512
```

The audit scans every selected year and writes `token_length_summary.json` with
separate comment and article-passage counts, percentages over the limit, p50/p95/p99,
and maxima. `truncated_records.parquet` contains only record type, year/month, stable
keys, token length, and a source-text SHA-256 checksum—never raw text. The normal
embedding run requires this audit and reuses a matching cached report. Its cache key
includes the tokenizer revision, source fingerprint, years, maximum length, and any
story limit.

CUDA and MPS embedding runs use adaptive batching by default. A recoverable
accelerator out-of-memory error clears the device cache, halves the batch, and retries
the same records without terminating the process or writing a partial article. After
25 successful calls that actually fill the current batch, it doubles cautiously
toward `--max-batch-size`; the ceiling defaults to the requested initial
`--batch-size`. For explicit upward probing, use for example:

```bash
python scripts/build_embeddings.py --device cuda \
  --batch-size 64 --max-batch-size 256 --min-batch-size 4
```

The progress output shows the current batch, and every increase/decrease is logged
and retained in the final manifest. Disable this behavior with
`--no-adaptive-batching`. Non-memory failures and OOM at the configured minimum still
stop the run; completed per-article checkpoints remain resumable.

Use repeated `--year` flags only when intentionally restricting a build, for
example `--year 2024 --year 2025`. Omitting `--year` is the safer default because
newly added year partitions are then included automatically. Annual QA summaries
are validated when present; a year collected only for selected months can use its
month-scoped QA summaries.

### Precomputed semantic similarities

After the complete embedding store finishes, calculate the three scalar semantic
features required by the preference models without loading BGE-M3 again:

```bash
python scripts/build_similarity_features.py \
  --model-id BAAI/bge-m3 \
  --revision 5617a9f61b028005a4858fdac845db406aefb181 \
  --year 2025 \
  --exact-novelty-threshold 5000 \
  --progress-every-stories 100
```

The resolver ignores watermarked smoke-test embedding stores and selects the one
compatible `COMPLETE_SOURCE` build. If more than one compatible full store exists,
pass its exact directory with `--embedding-store PATH`.

The command writes one resumable scalar checkpoint per story below
`model_output/selection_2025/similarities/`. It computes only:

* mean cosine similarity to the three closest article passages;
* novelty relative to strictly earlier comments; and
* root-comment novelty relative to strictly earlier roots.

For discussions with at most 5,000 eligible comments, one exact cosine matrix is
reused for both novelty definitions. Larger scopes use deterministic incremental
HNSW and are checked against exact strict-prior neighbours. The current collection
has eight 2025 discussions above 5,000 eligible comments; four also exceed 5,000
roots. The December 2024 lookback is embedded for reuse, but its similarity scores
are not calculated by default because they are not inputs to the 2025 models.

`06A_build_model_features.ipynb` resolves the completed similarity store and joins
these scalars using `story_id, comment_id`. It never reruns the embedding model or
recalculates cosine similarities. Set `COMMENTGAP_SIMILARITY_STORE` only when an
explicit build directory is needed; otherwise the compatible store is discovered
below `COMMENTGAP_SIMILARITY_ROOT`.
