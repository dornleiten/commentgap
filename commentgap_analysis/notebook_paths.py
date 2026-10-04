"""Run-scoped path setup shared by numbered notebook entry points."""

from __future__ import annotations

import os
from pathlib import Path

from commentgap_analysis.paths import ExecutionContext


# Notebook destinations.  Multi-output notebooks set their concrete sibling
# roots below; this prevents Stage 2 outputs being nested under features and
# keeps Stage 8 ranker inputs separate from its frozen winners.
STAGE_AREAS = {
    "01": "shared/raw_scrape",
    "02": "shared/features",
    "03": "shared/model_data",
    "04": "CG1/feature_diagnostics",
    "05": "CG1/descriptives",
    "06": "CG1/comment_gap",
    "08": "CG1/winners",
    "09": "CG1/reporting",
    "10": "CG2/forum",
    "11": "CG2/forum/inference",
    "12": "CG2/forum/reporting",
    "13": "CG2/ranking_similarity",
    "14": "CG2/topics",
    "15": "CG2/topics/analysis",
    "16": "CG2/topics/reporting",
}


# Keep input roots concrete.  In particular, hashing/protecting all of
# CG2/topics would prevent later stages writing analysis beside topic fits.
STAGE_INPUT_AREAS = {
    "02": ("shared/raw_scrape",),
    "03": ("shared/features",),
    "04": ("shared/features", "shared/model_data"),
    "05": ("shared/raw_scrape", "shared/model_data"),
    "06": ("shared/model_data", "CG1/descriptives"),
    "08": ("CG1/rankers/factorial",),
    "09": ("shared/model_data", "CG1/rankers/factorial", "CG1/winners", "CG1/regression"),
    "10": ("shared/model_data", "CG1/rankers/factorial", "CG1/winners", "CG1/regression", "shared/embeddings", "shared/raw_scrape"),
    "11": ("CG2/forum",),
    "12": ("CG2/forum/inference", "CG2/forum/policy_scores", "CG1/reporting", "CG1/regression"),
    "13": ("CG2/forum/policy_scores",),
    "14": ("shared/raw_scrape", "shared/embeddings", "shared/model_data"),
    "15": ("CG2/topics/runs", "CG2/topics/diagnostics"),
    "16": ("CG2/topics/runs", "CG2/topics/analysis"),
}


def notebook_run_root(
    stage: str,
    *,
    area: str | None = None,
    inputs: tuple[Path | str, ...] = (),
    context: ExecutionContext | None = None,
) -> Path:
    """Return a writable run-scoped root for a fresh or resumed notebook."""
    context = context or ExecutionContext.from_values()
    if context.mode == "frozen":
        raise RuntimeError("Frozen notebooks load saved inputs and have no writable analysis root")
    selected_area = area or STAGE_AREAS[stage]
    for input_path in inputs:
        context.read_path(input_path)
    root = context.prepare_run(
        contract={"workflow": "numbered-notebooks", "stage": stage},
        stage=selected_area,
    )
    return root / selected_area


def _configured_input(
    context: ExecutionContext,
    *,
    env_var: str,
    run_area: str | None = None,
    key: str | None = None,
) -> Path:
    """Resolve one input, preserving an explicit environment override."""
    explicit = os.environ.get(env_var)
    if explicit:
        resolved = context.read_root(key, explicit=explicit) if key else context.read_path(explicit)
    else:
        candidate = context.run_path(run_area) if run_area else None
        if candidate is not None and candidate.exists():
            resolved = context.read_path(candidate)
        elif key is not None and key in context.paths.entries:
            resolved = context.read_root(key)
        elif key is not None:
            # Keep small standalone fixtures usable when they configure only
            # ``outputs``; a real repository always has the named entry.
            resolved = context.read_path(context.paths.root / key)
        elif candidate is not None:
            resolved = context.read_path(candidate)
        else:
            raise ValueError(f"No source configured for {env_var}")
        os.environ[env_var] = str(resolved)
    return resolved


def _output(context: ExecutionContext, env_var: str, area: str) -> Path:
    path = context.run_path(area)
    os.environ[env_var] = str(path)
    return path


def _configured_file(
    context: ExecutionContext,
    *,
    env_var: str,
    filename: str,
    run_area: str | None = None,
    key: str | None = None,
    relative_path: str | None = None,
) -> Path:
    """Resolve a file input while retaining its concrete path in provenance."""
    explicit = os.environ.get(env_var)
    if explicit:
        resolved = context.read_path(explicit)
    else:
        candidate = context.run_path(run_area) / filename if run_area else None
        if candidate is not None and candidate.exists():
            resolved = context.read_path(candidate)
        elif key is not None and key in context.paths.entries:
            base = context.read_root(key)
            if base.is_dir() and base in context.read_inputs:
                context.read_inputs.remove(base)
            resolved = base if base.is_file() else context.read_path(base / (relative_path or filename))
        elif candidate is not None:
            resolved = context.read_path(candidate)
        else:
            resolved = context.read_path(context.paths.root / filename)
        os.environ[env_var] = str(resolved)
    return resolved


def _set_output(env_var: str, path: Path) -> Path:
    os.environ[env_var] = str(path)
    return path


def _sources(context: ExecutionContext, stage: str) -> list[Path]:
    """Configure input variables and return paths to hash in the stage manifest."""
    inputs: list[Path] = []

    def add(path: Path) -> None:
        path = context.read_path(path)
        if path not in inputs:
            inputs.append(path)

    if stage == "01":
        return inputs
    if stage == "02":
        add(_configured_input(context, env_var="COMMENTGAP_DATA_ROOT", run_area="shared/raw_scrape", key="raw_scrape"))
        # Embeddings, similarities, and AQuA are sibling producer outputs in
        # this notebook.  Explicit values remain inputs; otherwise their
        # producer roots are assigned after the manifest is prepared.
        for env_var, key, area in (("COMMENTGAP_EMBEDDING_ROOT", "embeddings", "shared/embeddings"), ("COMMENTGAP_SIMILARITY_ROOT", "similarities", "shared/similarities"), ("COMMENTGAP_AQUA_ROOT", "aqua", "shared/aqua")):
            explicit = os.environ.get(env_var)
            # Values installed by an earlier configure call are producer
            # destinations, not new immutable inputs on resume.
            is_internal_output = explicit and Path(explicit).resolve() == context.run_path(area).resolve()
            if explicit and not is_internal_output:
                add(_configured_input(context, env_var=env_var, key=key))
        _set_output("COMMENTGAP_MODEL_ROOT", context.run_path("shared"))
        _set_output("COMMENTGAP_EMBEDDING_ROOT", context.run_path("shared/embeddings"))
        _set_output("COMMENTGAP_SIMILARITY_ROOT", context.run_path("shared/similarities"))
        _set_output("COMMENTGAP_AQUA_ROOT", context.run_path("shared/aqua"))
        _set_output("COMMENTGAP_FEATURE_ROOT", context.run_path("shared/features"))
        return inputs
    if stage == "03":
        add(_configured_input(context, env_var="COMMENTGAP_FEATURE_ROOT", run_area="shared/features", key="features"))
    elif stage == "04":
        add(_configured_input(context, env_var="COMMENTGAP_FEATURE_ROOT", run_area="shared/features", key="features"))
        add(_configured_input(context, env_var="COMMENTGAP_MODEL_DATA_ROOT", run_area="shared/model_data", key="model_data"))
    elif stage == "05":
        add(_configured_input(context, env_var="COMMENTGAP_MODEL_DATA_ROOT", run_area="shared/model_data", key="model_data"))
        add(_configured_input(context, env_var="COMMENTGAP_DATA_ROOT", run_area="shared/raw_scrape", key="raw_scrape"))
    elif stage == "06":
        add(_configured_input(context, env_var="COMMENTGAP_MODEL_DATA_ROOT", run_area="shared/model_data", key="model_data"))
        add(_configured_input(context, env_var="COMMENTGAP_DESCRIPTIVES_ROOT", run_area="CG1/descriptives", key="frozen_cg1_descriptives"))
    elif stage == "08":
        add(_configured_input(context, env_var="COMMENTGAP_FACTORIAL_ROOT", run_area="CG1/rankers/factorial", key="frozen_cg1_factorial_rankers"))
    elif stage == "09":
        add(_configured_input(context, env_var="COMMENTGAP_MODEL_DATA_ROOT", run_area="shared/model_data", key="model_data"))
        add(_configured_input(context, env_var="COMMENTGAP_FACTORIAL_ROOT", run_area="CG1/rankers/factorial", key="frozen_cg1_factorial_rankers"))
        add(_configured_input(context, env_var="COMMENTGAP_FACTORIAL_WINNER_ROOT", run_area="CG1/winners", key="frozen_cg1_winners"))
        add(_configured_input(context, env_var="COMMENTGAP_REGRESSION_ROOT", run_area="CG1/regression", key="frozen_cg1_regression"))
    elif stage == "10":
        add(_configured_input(context, env_var="COMMENTGAP_MODEL_DATA_ROOT", run_area="shared/model_data", key="model_data"))
        add(_configured_input(context, env_var="COMMENTGAP_FACTORIAL_ROOT", run_area="CG1/rankers/factorial", key="frozen_cg1_factorial_rankers"))
        add(_configured_input(context, env_var="COMMENTGAP_FACTORIAL_WINNER_ROOT", run_area="CG1/winners", key="frozen_cg1_winners"))
        add(_configured_input(context, env_var="COMMENTGAP_REGRESSION_ROOT", run_area="CG1/regression", key="frozen_cg1_regression"))
        add(_configured_input(context, env_var="COMMENTGAP_DATA_ROOT", run_area="shared/raw_scrape", key="raw_scrape"))
        embedding_root = _configured_input(context, env_var="COMMENTGAP_EMBEDDING_ROOT", run_area="shared/embeddings", key="embeddings")
        embedding_store = Path(os.environ.get("COMMENTGAP_EMBEDDING_STORE", str(embedding_root / "model=BAAI__bge-m3--d790e737" / "build=de5b3016fb2f-010a7cc75a88")))
        os.environ["COMMENTGAP_EMBEDDING_STORE"] = str(embedding_store)
        add(embedding_store)
    elif stage == "11":
        forum_root = _configured_input(context, env_var="COMMENTGAP_FORUM_ANALYSIS_ROOT", run_area="CG2/forum", key="frozen_cg2_forum")
        if forum_root in context.read_inputs:
            context.read_inputs.remove(forum_root)
        add(forum_root / "analysis_comments.parquet")
        add(forum_root / "policy_scores" / "policy_scores.parquet")
    elif stage == "12":
        add(_configured_input(context, env_var="COMMENTGAP_FORUM_INFERENCE_ROOT", run_area="CG2/forum/inference", key="frozen_cg2_forum"))
        forum_root = _configured_input(context, env_var="COMMENTGAP_FORUM_ANALYSIS_ROOT", run_area="CG2/forum", key="frozen_cg2_forum")
        if forum_root in context.read_inputs:
            context.read_inputs.remove(forum_root)
        add(forum_root / "policy_scores" / "policy_scores.parquet")
        report_root = _configured_input(context, env_var="COMMENTGAP_REPORT_ROOT", run_area="CG1/reporting", key="frozen_cg1_reporting")
        add(report_root)
        os.environ["COMMENTGAP_REPORT_TABLES_ROOT"] = str(report_root / "tables")
        add(_configured_input(context, env_var="COMMENTGAP_REGRESSION_ROOT", run_area="CG1/regression", key="frozen_cg1_regression"))
    elif stage == "13":
        add(_configured_file(context, env_var="COMMENTGAP_FORUM_SCORES_PATH", filename="policy_scores.parquet", run_area="CG2/forum/policy_scores", key="frozen_cg2_forum", relative_path="policy_scores/policy_scores.parquet"))
    elif stage == "14":
        add(_configured_input(context, env_var="COMMENTGAP_DATA_ROOT", run_area="shared/raw_scrape", key="raw_scrape"))
        embedding_root = _configured_input(context, env_var="COMMENTGAP_EMBEDDING_ROOT", run_area="shared/embeddings", key="embeddings")
        embedding_store = Path(os.environ.get("COMMENTGAP_EMBEDDING_STORE", str(embedding_root / "model=BAAI__bge-m3--d790e737" / "build=de5b3016fb2f-010a7cc75a88")))
        os.environ["COMMENTGAP_EMBEDDING_STORE"] = str(embedding_store)
        add(context.read_path(embedding_store))
        add(_configured_file(context, env_var="COMMENTGAP_SPLIT_PATH", filename="master_article_split.parquet", run_area="shared/model_data", key="source_split"))
        add(_configured_file(context, env_var="COMMENTGAP_ANALYSIS_COMMENTS_PATH", filename="analysis_comments.parquet", run_area="CG2/forum", key="frozen_cg2_topic_source"))
    elif stage == "15":
        add(_configured_input(context, env_var="COMMENTGAP_TOPIC_RUNS_ROOT", run_area="CG2/topics/runs", key="frozen_cg2_topics_runs"))
        add(_configured_input(context, env_var="COMMENTGAP_TOPIC_DIAGNOSTICS_ROOT", run_area="CG2/topics/diagnostics", key="frozen_cg2_topics_diagnostics"))
        add(_configured_file(context, env_var="COMMENTGAP_ANALYSIS_COMMENTS_PATH", filename="analysis_comments.parquet", run_area="CG2/forum", key="frozen_cg2_topic_source"))
        add(_configured_file(context, env_var="COMMENTGAP_SPLIT_PATH", filename="master_article_split.parquet", run_area="shared/model_data", key="source_split"))
        add(_configured_file(context, env_var="COMMENTGAP_CHOICE_SET_PATH", filename="choice_set_all.parquet", run_area="shared/model_data", key="model_data"))
        add(_configured_input(context, env_var="COMMENTGAP_DATA_ROOT", run_area="shared/raw_scrape", key="raw_scrape") / "comments" / "year=2025")
    elif stage == "16":
        topic_runs = _configured_input(context, env_var="COMMENTGAP_TOPIC_RUNS_ROOT", run_area="CG2/topics/runs", key="frozen_cg2_topics_runs")
        add(topic_runs)
        configured_analysis = context.run_path("CG2/topics/analysis")
        if context.mode == "frozen" or not configured_analysis.exists():
            analysis_root = topic_runs / "mcs15_nn10_ms5_seed2026_1e1b8fce5deb3e58" / "analysis"
            os.environ["COMMENTGAP_TOPIC_INPUT_ROOT"] = str(analysis_root)
            add(analysis_root)
        else:
            add(_configured_input(context, env_var="COMMENTGAP_TOPIC_INPUT_ROOT", run_area="CG2/topics/analysis", key="frozen_cg2_topics_runs"))
        add(_configured_file(context, env_var="COMMENTGAP_ANALYSIS_COMMENTS_PATH", filename="analysis_comments.parquet", run_area="CG2/forum", key="frozen_cg2_topic_source"))
    else:
        raise KeyError(stage)
    return inputs


def configure_notebook_stage(stage: str) -> Path:
    """Configure a notebook and return its run-scoped producer destination."""
    if stage not in STAGE_AREAS:
        raise KeyError(stage)
    context = ExecutionContext.from_values()
    explicit_stage2_inputs = {
        name: os.environ.get(name)
        for name in ("COMMENTGAP_EMBEDDING_ROOT", "COMMENTGAP_SIMILARITY_ROOT", "COMMENTGAP_AQUA_ROOT")
    }
    stage_inputs = tuple(_sources(context, stage))
    area = STAGE_AREAS[stage]
    # The notebook is an orchestrator: its manifest entry must not collide
    # with a subprocess CLI that later prepares the concrete output area.
    run_root = context.prepare_run(
        contract={"workflow": "numbered-notebooks", "stage": stage},
        stage=f"notebook/{stage}",
    )
    output = run_root / area

    if stage == "01":
        _set_output("COMMENTGAP_DATA_ROOT", output)
    elif stage == "02":
        _set_output("COMMENTGAP_MODEL_ROOT", context.run_path("shared"))
        for name, area in (("COMMENTGAP_EMBEDDING_ROOT", "shared/embeddings"), ("COMMENTGAP_SIMILARITY_ROOT", "shared/similarities"), ("COMMENTGAP_AQUA_ROOT", "shared/aqua")):
            if explicit_stage2_inputs[name]:
                os.environ[name] = explicit_stage2_inputs[name]
            else:
                _set_output(name, context.run_path(area))
        _set_output("COMMENTGAP_FEATURE_ROOT", output)
    elif stage == "03":
        _set_output("COMMENTGAP_MODEL_DATA_ROOT", output)
    elif stage == "04":
        _set_output("COMMENTGAP_FEATURE_DIAGNOSTIC_ROOT", output)
    elif stage == "05":
        _set_output("COMMENTGAP_DESCRIPTIVES_ROOT", output)
    elif stage == "06":
        _set_output("COMMENTGAP_COMMENT_GAP_ROOT", output)
    elif stage == "08":
        _set_output("COMMENTGAP_FACTORIAL_WINNER_ROOT", output)
    elif stage == "09":
        _set_output("COMMENTGAP_REPORT_ROOT", output)
        _set_output("COMMENTGAP_XGBOOST_CACHE_ROOT", output / "cache" / "xgboost_bge")
    elif stage == "10":
        _set_output("COMMENTGAP_FORUM_ANALYSIS_ROOT", output)
        _set_output("COMMENTGAP_FORUM_HANDOFF_ROOT", output / "ranker_handoff")
        _set_output("COMMENTGAP_FORUM_POLICY_ROOT", output / "policy_scores")
        _set_output("COMMENTGAP_FORUM_INFERENCE_ROOT", output / "inference")
        _set_output("COMMENTGAP_FORUM_REPORTING_ROOT", output / "reporting")
    elif stage == "11":
        _set_output("COMMENTGAP_FORUM_ANALYSIS_ROOT", context.run_path("CG2/forum"))
        _set_output("COMMENTGAP_FORUM_INFERENCE_ROOT", output)
        _set_output("COMMENTGAP_FORUM_REPORTING_ROOT", context.run_path("CG2/forum/reporting"))
    elif stage == "12":
        _set_output("COMMENTGAP_FORUM_REPORTING_ROOT", output)
        _set_output("COMMENTGAP_FORUM_SCORES_PATH", context.run_path("CG2/forum/policy_scores/policy_scores.parquet"))
    elif stage == "13":
        _set_output("COMMENTGAP_RANKING_SIMILARITY_ROOT", output)
    elif stage == "14":
        _set_output("COMMENTGAP_TOPIC_RUNS_ROOT", context.run_path("CG2/topics/runs"))
        _set_output("COMMENTGAP_TOPIC_DIAGNOSTICS_ROOT", context.run_path("CG2/topics/diagnostics"))
        _set_output("COMMENTGAP_TOPIC_ANALYSIS_ROOT", context.run_path("CG2/topics/analysis"))
    elif stage == "15":
        _set_output("COMMENTGAP_TOPIC_ANALYSIS_ROOT", output)
    elif stage == "16":
        _set_output("COMMENTGAP_TOPIC_INPUT_ROOT", context.run_path("CG2/topics/analysis"))
        _set_output("COMMENTGAP_TOPIC_ANALYSIS_ROOT", output)
    return output
