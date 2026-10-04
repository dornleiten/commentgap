# CommentGap

CommentGap studies a 2025 collection of news-article discussions. Two linked analyses use the same collected data:

- **CG1:** how audience and journalist comment preferences differ, and how well selection models predict them.
- **CG2:** how ranking policies affect forum outcomes and the topics readers encounter.

The [CG1](submissions/CG1/README.md) and [CG2](submissions/CG2/README.md) submissions are preserved with their figures and tables. Their source mapping and checksums are in [paper-assets.csv](provenance/paper-assets.csv).

## Run the notebooks

Install the Python environment from the repository root:

```bash
python -m pip install -r requirements.txt
python -m pip install -e .
python scripts/canonical_artifacts.py verify
jupyter lab
```

The first code cell of each notebook sets `mode = "replay"`. This mode reads the compact public inputs in [artifacts/canonical](artifacts/canonical/) and regenerates the displayed tables and figures. Set `mode = "recompute"` to use the larger private saved inputs, or `"fresh"` to run the scientific producers. Those modes require data and models that are not in the public repository. Notebook outputs go under `outputs/`.

The three R Markdown reports also default to Replay. Their packages are recorded in [renv.lock](renv.lock); after restoring that R environment, render them with:

```bash
Rscript scripts/render_rmd.R 07_stacked_selection_models.Rmd 07A1_collinearity_sensitivity.Rmd 07A2_efron_exact_sensitivity.Rmd
```

## Notebook and report guide

Run stages in number order for a Fresh workflow. Replay can be opened one stage at a time.

| Stage | File | Purpose |
| --- | --- | --- |
| 01 | [Scrape discussions](01_scrape_discussions.ipynb) | Collect and summarize articles, forums, and comments. |
| 02 | [Build model features](02_build_model_features.ipynb) | Prepare embeddings and features for the models. |
| 03 | [Shared preprocessing](03_shared_model_preprocessing.ipynb) | Create the article split and model-ready inputs. |
| 04 | [Feature diagnostics](04_feature_distribution_diagnostics.ipynb) | Check feature distributions and correlations. |
| 05 | [Descriptive statistics](05_descriptive_statistics_topics.ipynb) | Summarize the collection and its topic breakdown. |
| 06 | [Comment-gap score](06_comment_gap_score.ipynb) | Calculate and describe the comment-gap measure. |
| 07 | [Stacked selection models](07_stacked_selection_models.Rmd) | Fit and report audience–journalist selection models. |
| 07A1 | [Collinearity sensitivity](07A1_collinearity_sensitivity.Rmd) | Check how correlated predictors affect those models. |
| 07A2 | [Tie-method sensitivity](07A2_efron_exact_sensitivity.Rmd) | Compare Efron and exact conditional-logit results. |
| 08 | [Model variants](08_xgboost_neural_model_variants.ipynb) | Compare XGBoost and neural rankers; select development winners. |
| 09 | [CG1 tables and plots](09_model_tables_plots.ipynb) | Present model performance and feature effects. |
| 10 | [FORUM scores](10_calculate_forum_scores.ipynb) | Build or load forum outcomes for ranking policies. |
| 11 | [Ranking effects](11_ranking_algorithm_effects.ipynb) | Compare ordering, reply, and pinning choices. |
| 12 | [FORUM correlations](12_forum_correlations.ipynb) | Compare forum outcomes with ranking metrics. |
| 13 | [Ranking similarity](13_ranking_algorithm_similarity.ipynb) | Compare and cluster ranking policies. |
| 14 | [Topic model](14_topic_model_fit.ipynb) | Fit and assess the shared article/comment topic model. |
| 15 | [Topic calculations](15_topic_agenda_calculations.ipynb) | Calculate topic exposure and vote-attention results. |
| 16 | [Topic results](16_topic_agenda_analysis.ipynb) | Present topic coverage and ranking effects. |

Code supporting the notebooks is in `commentgap_analysis/` and `commentgap_scraper/`. The [script index](scripts/README.md) distinguishes current commands from historical reconstruction methods. `data/`, private saved products, and generated `outputs/` are excluded from Git.
