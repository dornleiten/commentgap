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

Use repeated `--story-id ID` arguments to target known fixtures or unusual forums.
Outputs default to `data/scrape_2025/`, which is ignored by Git. The main tables are
partitioned Parquet datasets named `articles`, `forums`, `forum_pages`, and `comments`; the SQLite
manifest and page-level staging files make interrupted crawls resumable. Validation
writes both JSON and Parquet summaries under `qa_summary`.

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
