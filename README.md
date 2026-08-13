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
commentgap-scrape crawl --year 2025 --limit 100 --selection monthly-round-robin
commentgap-scrape validate --year 2025 --allow-incomplete

# Resume pending/in-progress work; add --retry-failed to revisit failures.
commentgap-scrape crawl --year 2025
commentgap-scrape crawl --year 2025 --retry-failed

# Write normalized compatibility tables for the existing R analysis.
commentgap-scrape export-legacy --year 2025
```

Use repeated `--story-id ID` arguments to target known fixtures or unusual forums.
Outputs default to `data/scrape_2025/`, which is ignored by Git. The main tables are
partitioned Parquet datasets named `articles`, `forums`, and `comments`; the SQLite
manifest and page-level staging files make interrupted crawls resumable. Validation
writes both JSON and Parquet summaries under `qa_summary`.

Progress is written to the normal log stream so it remains readable in an interactive
terminal, a redirected log file, or an unattended job. Crawl updates include overall
percentage, elapsed time, ETA, throughput, and status counts. Large forums also report
page and posting progress every ten GraphQL pages; adjust this with
`--progress-every-pages N`.

The compatibility export intentionally keeps article text in a separate article
table rather than duplicating it for every comment. `user_names` contains the stable
pseudonym, and timestamp columns use ISO 8601 UTC values.

Unauthenticated article responses do not always expose a publication timestamp. In
that case, `published_at` uses the sitemap's `lastmod` value and
`published_at_source` is `sitemap_lastmod_fallback`. Because a sitemap timestamp can
also reflect an article update, analyses that depend on exact publication time should
filter or sensitivity-check this field. The original sitemap value is retained in
`sitemap_lastmod` for auditability.
