"""Fit BERTopic on article documents and transform article/comment documents in resumable batches."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import importlib.metadata
import json
import os
from platform import platform, python_version
from pathlib import Path
import time
import subprocess

import duckdb
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from commentgap_analysis.topic_modeling import (
    TopicModelConfig,
    fit_topic_model,
    load_precomputed_embeddings,
    load_topic_model,
    prepare_article_passages,
    prepare_documents,
    save_topic_model,
)
from commentgap_analysis.topic_modeling import topic_model_quality_table


from commentgap_analysis.topic_runs import (
    complete_run, file_inventory, prepare_run, resolve_run,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Fit the Paper 2 BERTopic model with automatic topic reduction.")
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument("--split", type=Path, default=None)
    parser.add_argument("--analysis-comments", type=Path, default=None, help="Paper 2 comment sample used for transformation (defaults to paper2/analysis_comments.parquet).")
    output = parser.add_mutually_exclusive_group()
    output.add_argument("--output-root", type=Path, default=None)
    output.add_argument("--runs-root", type=Path, default=None,
                        help="Store each setup in its own registered run directory.")
    parser.add_argument("--model-config", type=Path, default=None,
                        help="JSON object of TopicModelConfig fields; defaults for omitted fields.")
    parser.add_argument("--run-tag", default="",
                        help="Optional replicate label to create a separate run of the same setup.")
    parser.add_argument("--embedding-store", type=Path, default=None, help="Completed BGE-M3 embedding build directory.")
    parser.add_argument("--year", type=int, default=2025)
    parser.add_argument("--fit-role", choices=["development", "paper2_test"], default="development")
    parser.add_argument("--all-stories", action="store_true", help="Fit on all stories instead of development stories; transformations remain restricted to Paper 2 test stories.")
    fit_comments = parser.add_mutually_exclusive_group()
    fit_comments.add_argument("--fit-comment-documents", type=Path, default=None,
                              help="Exact prepared comment sample exported by notebook 14.")
    fit_comments.add_argument(
        "--include-comments-in-fit",
        action="store_true",
        help="Use comments as well as articles when fitting; default fits articles only.",
    )
    parser.add_argument(
        "--calculate-probabilities",
        action="store_true",
        help="Retain BERTopic's HDBSCAN soft topic memberships for every document.",
    )
    parser.add_argument("--batch-size", type=int, default=10_000, help="Documents per transformation batch.")
    parser.add_argument("--workers", type=int, default=os.cpu_count() or 1, help="Concurrent test-transformation workers (default: all logical CPUs).")
    return parser


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _code_revision(repo_root: Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(repo_root), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _package_versions() -> dict[str, str]:
    result = {}
    for name in ("bertopic", "umap-learn", "hdbscan", "sentence-transformers", "pandas", "numpy", "pyarrow", "scikit-learn", "scipy", "numba", "duckdb", "joblib"):
        try:
            result[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            result[name] = "not-installed"
    return result


def _coverage_rows(coverage: dict[str, dict]) -> pd.DataFrame:
    rows = []
    for story_id, values in sorted(coverage.items()):
        rows.append(
            {
                "story_id": story_id,
                "n_sections": values["n_sections"],
                "n_valid_sections": values["n_valid_sections"],
                "n_outlier_sections": values["n_outlier_sections"],
                "fraction_sections_valid_topic": values["n_valid_sections"] / values["n_sections"] if values["n_sections"] else 0.0,
                "n_unique_topics": len(values["topics"]),
                "has_valid_topic": values["n_valid_sections"] > 0,
                "n_documents": values["n_documents"],
                "n_valid_documents": values["n_valid_documents"],
                "n_comment_documents": values["n_comment_documents"],
                "n_valid_comment_documents": values["n_valid_comment_documents"],
                "article_topic_assignment": -1,
                "article_has_valid_topic": values["n_valid_sections"] > 0,
            }
        )
    return pd.DataFrame(
        rows,
        columns=[
            "story_id",
            "n_sections",
            "n_valid_sections",
            "n_outlier_sections",
            "fraction_sections_valid_topic",
            "n_unique_topics",
            "has_valid_topic",
            "n_documents",
            "n_valid_documents",
            "n_comment_documents",
            "n_valid_comment_documents",
            "article_topic_assignment",
            "article_has_valid_topic",
        ],
    )


def _update_coverage(coverage: dict[str, dict], memberships: pd.DataFrame) -> None:
    for story_id, group in memberships.groupby("story_id", sort=False):
        story_id = str(story_id)
        values = coverage.setdefault(
            story_id,
            {
                "n_sections": 0,
                "n_valid_sections": 0,
                "n_outlier_sections": 0,
                "topics": set(),
                "n_documents": 0,
                "n_valid_documents": 0,
                "n_comment_documents": 0,
                "n_valid_comment_documents": 0,
            },
        )
        valid = group["valid_topic"].astype(bool)
        is_article = group["doc_type"].eq("article")
        is_comment = group["doc_type"].eq("comment")
        article_valid = valid & is_article
        values["n_sections"] += int(is_article.sum())
        values["n_valid_sections"] += int(article_valid.sum())
        values["n_outlier_sections"] += int((is_article & ~valid).sum())
        values["topics"].update(group.loc[article_valid, "topic_assignment"].astype(int).tolist())
        values["n_documents"] += len(group)
        values["n_valid_documents"] += int(valid.sum())
        values["n_comment_documents"] += int(is_comment.sum())
        values["n_valid_comment_documents"] += int((is_comment & valid).sum())


def _coverage_for_json(coverage: dict[str, dict]) -> dict[str, dict]:
    return {
        story_id: {**values, "topics": sorted(values["topics"])}
        for story_id, values in coverage.items()
    }


def _coverage_from_json(raw: dict) -> dict[str, dict]:
    result = {}
    for story_id, values in raw.items():
        result[str(story_id)] = {**values, "topics": set(values.get("topics", []))}
    return result


def _write_progress(path: Path, progress: dict) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(progress, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _empty_articles() -> pd.DataFrame:
    return pd.DataFrame(columns=["story_id", "title", "subtitle", "body"])


def _empty_comments() -> pd.DataFrame:
    return pd.DataFrame(columns=["story_id", "comment_id", "effective_text", "text"])


def _ensure_document_schema(documents: pd.DataFrame) -> pd.DataFrame:
    documents = documents.copy()
    if "comment_id" not in documents.columns:
        documents["comment_id"] = pd.Series(pd.NA, index=documents.index, dtype="string")
    for column in ("doc_id", "story_id", "comment_id", "doc_type"):
        documents[column] = documents[column].astype("string")
    return documents


def _table_with_schema(memberships: pd.DataFrame, schema: pa.Schema | None = None) -> pa.Table:
    table = pa.Table.from_pandas(memberships, preserve_index=False)
    if schema is not None:
        table = table.cast(schema)
    return table


def _report_progress(label: str, processed: int, total: int, started: float, batch_number: int) -> None:
    elapsed = time.perf_counter() - started
    rate = processed / elapsed if elapsed > 0 else 0.0
    remaining = max(0, total - processed)
    eta_seconds = remaining / rate if rate > 0 else float("nan")
    eta = f"{eta_seconds / 60:.1f} min" if eta_seconds == eta_seconds else "n/a"
    percentage = 100.0 * processed / total if total else 100.0
    print(
        f"[{label}] batch {batch_number}: {processed:,}/{total:,} "
        f"({percentage:.1f}%), {rate:,.0f} documents/s, elapsed {elapsed / 60:.1f} min, ETA {eta}.",
        flush=True,
    )


def _merge_batch_files(batch_root: Path, output_path: Path) -> None:
    batch_files = sorted(batch_root.glob("*.parquet"))
    if not batch_files:
        raise FileNotFoundError(f"No completed membership batches found in {batch_root}")
    temporary = output_path.with_suffix(".partial.parquet")
    writer = None
    try:
        output_schema = None
        for batch_path in batch_files:
            table = pq.read_table(batch_path)
            if writer is None:
                output_schema = table.schema
                writer = pq.ParquetWriter(temporary, output_schema)
            elif table.schema != output_schema:
                columns = {field.name: table[field.name] for field in table.schema}
                arrays = []
                for field in output_schema:
                    array = columns.get(field.name)
                    if array is None:
                        array = pa.nulls(table.num_rows, type=field.type)
                    elif not array.type.equals(field.type):
                        array = array.cast(field.type, safe=False)
                    arrays.append(array)
                table = pa.Table.from_arrays(arrays, schema=output_schema)
            writer.write_table(table)
    finally:
        if writer is not None:
            writer.close()
    temporary.replace(output_path)


def _transform_batch(bundle, documents: pd.DataFrame, embedding_store: Path) -> pd.DataFrame:
    """Transform one independent batch; called from worker threads."""
    return bundle.transform(
        documents,
        embeddings=load_precomputed_embeddings(documents, embedding_store),
    )


def _commit_transformed_batch(
    batch_root: Path,
    batch_number: int,
    batch_prefix: str,
    memberships: pd.DataFrame,
    progress: dict,
    coverage: dict[str, dict],
    started: float,
    total: int,
    progress_prefix: str,
    progress_label: str,
) -> None:
    batch_path = batch_root / f"{batch_prefix}_{batch_number:06d}.parquet"
    temporary = batch_path.with_suffix(".tmp.parquet")
    pq.write_table(_table_with_schema(memberships), temporary)
    temporary.replace(batch_path)

    _update_coverage(coverage, memberships)
    batch_key = f"{progress_prefix}_batches"
    done_key = f"{progress_prefix}_done"
    progress[batch_key].append(batch_number)
    progress[done_key] += len(memberships)
    progress["coverage"] = _coverage_for_json(coverage)
    _write_progress(batch_root / "progress.json", progress)
    _report_progress(progress_label, progress[done_key], total, started, batch_number)


def _parallel_transform_batches(
    reader,
    bundle,
    embedding_store: Path,
    batch_root: Path,
    progress: dict,
    coverage: dict[str, dict],
    batch_prefix: str,
    total: int,
    started: float,
    batch_size: int,
    workers: int,
    prepare,
    progress_prefix: str,
    progress_label: str,
) -> None:
    """Transform document chunks concurrently while committing results in the main thread."""
    from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait

    pending = {}
    completed = set(progress.get(f"{progress_prefix}_batches", []))
    next_batch_number = 0

    def commit_done(done) -> None:
        for future in done:
            batch_number = pending.pop(future)
            _commit_transformed_batch(
                batch_root,
                batch_number,
                batch_prefix,
                future.result(),
                progress,
                coverage,
                started,
                total,
                progress_prefix,
                progress_label,
            )

    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix=f"topic-{batch_prefix}") as executor:
        for source_batch in reader:
            source_documents = _ensure_document_schema(prepare(source_batch.to_pandas()))
            for start in range(0, len(source_documents), batch_size):
                next_batch_number += 1
                batch_number = next_batch_number
                if batch_number in completed:
                    continue
                document_chunk = source_documents.iloc[start : start + batch_size].copy()
                future = executor.submit(_transform_batch, bundle, document_chunk, embedding_store)
                pending[future] = batch_number
                if len(pending) >= workers:
                    done, _ = wait(pending, return_when=FIRST_COMPLETED)
                    commit_done(done)

        while pending:
            done, _ = wait(pending, return_when=FIRST_COMPLETED)
            commit_done(done)

def _model_config(args: argparse.Namespace) -> TopicModelConfig:
    config_values = json.loads(args.model_config.read_text()) if args.model_config else {}
    if "ngram_range" in config_values:
        config_values["ngram_range"] = tuple(config_values["ngram_range"])
    if args.calculate_probabilities:
        config_values["calculate_probabilities"] = True
    model_config = TopicModelConfig(**config_values)
    if model_config.model_type != "bertopic":
        raise ValueError("This runner requires a BERTopic configuration")
    return model_config


def _load_fit_comment_documents(path: Path, split_path: Path, fit_role="development") -> pd.DataFrame:
    documents = pd.read_parquet(path)
    required = {'doc_id', 'story_id', 'comment_id', 'doc_type', 'text'}
    if not required.issubset(documents.columns) or documents.empty:
        raise ValueError('Fit comment sample must contain nonempty prepared comment documents')
    if documents[list(required)].isna().any().any():
        raise ValueError('Fit comment sample has missing identifiers or text')
    for column in ('doc_id', 'story_id', 'comment_id'):
        documents[column] = documents[column].astype(str)
    expected_ids = 'comment:' + documents.story_id + ':' + documents.comment_id
    if (not documents.doc_id.is_unique or not documents.doc_type.eq('comment').all()
            or not documents.doc_id.eq(expected_ids).all()
            or documents.text.astype(str).str.strip().eq('').any()):
        raise ValueError('Invalid or duplicate prepared fit comments')
    split = pd.read_parquet(split_path, columns=['story_id', 'split_role'])
    allowed = set(split.loc[split.split_role.eq(fit_role), 'story_id'].astype(str))
    if not documents.story_id.isin(allowed).all():
        raise ValueError(f'Fit comment sample must contain {fit_role} stories only')
    return documents.sort_values('doc_id').reset_index(drop=True)


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.fit_comment_documents and args.all_stories:
        raise ValueError('The sampled-comment variant requires development-only fitting')
    model_config = _model_config(args)
    args.calculate_probabilities = model_config.calculate_probabilities
    if args.batch_size < 1:
        raise ValueError("--batch-size must be positive")
    if args.workers < 1:
        raise ValueError("--workers must be positive")
    repo_root = args.repo_root.resolve()
    data_root = (args.data_root or repo_root / "data/scrape_2025").resolve()
    split_path = (args.split or repo_root / "model_output/selection_2025/model_data/master_article_split.parquet").resolve()
    analysis_comments_path = (args.analysis_comments or repo_root / "model_output/selection_2025/paper2/analysis_comments.parquet").resolve()
    output_root = (args.output_root or repo_root / "model_output/selection_2025/paper2_topic_modeling").resolve()
    embedding_store = (args.embedding_store or repo_root / "model_output/selection_2025/embeddings/model=BAAI__bge-m3--d790e737/build=de5b3016fb2f-010a7cc75a88").resolve()

    if not split_path.exists():
        raise FileNotFoundError(f"Required Paper 2 split file not found: {split_path}")
    if not analysis_comments_path.exists():
        raise FileNotFoundError(f"Required Paper 2 analysis-comments file not found: {analysis_comments_path}")

    development_filter = (
        " AND CAST(story_id AS VARCHAR) IN "
        "(SELECT CAST(story_id AS VARCHAR) FROM read_parquet(?) WHERE split_role = 'development')"
    )
    development_params: list[object] = [str(split_path)]
    test_filter = (
        " AND CAST(story_id AS VARCHAR) IN "
        "(SELECT CAST(story_id AS VARCHAR) FROM read_parquet(?) WHERE split_role = 'paper2_test')"
    )
    test_params: list[object] = [str(split_path)]
    role_filter = test_filter if args.fit_role == "paper2_test" else development_filter
    fit_filter = "" if args.all_stories else role_filter
    fit_params: list[object] = [] if args.all_stories else development_params

    article_sql = (
        "SELECT CAST(story_id AS VARCHAR) AS story_id, title, subtitle, body "
        "FROM read_parquet(?, hive_partitioning=true, union_by_name=true) "
        "WHERE year = ?" + fit_filter
    )
    comment_sql = (
        "SELECT CAST(story_id AS VARCHAR) AS story_id, CAST(comment_id AS VARCHAR) AS comment_id, effective_text, text "
        "FROM read_parquet(?, hive_partitioning=true, union_by_name=true) "
        "WHERE year = ? AND lower(COALESCE(lifecycle_status, 'published')) = 'published'" + fit_filter
        + " ORDER BY CAST(story_id AS VARCHAR), CAST(comment_id AS VARCHAR)"
    )

    con = duckdb.connect()
    try:
        articles = con.execute(article_sql, [str(data_root / "articles/**/*.parquet"), args.year, *fit_params]).df()
        comments = (
            con.execute(comment_sql, [str(data_root / "comments/**/*.parquet"), args.year, *fit_params]).df()
            if args.include_comments_in_fit
            else _empty_comments()
        )
    finally:
        con.close()

    if args.fit_comment_documents:
        comments = _load_fit_comment_documents(args.fit_comment_documents, split_path, args.fit_role)
        training_documents = _ensure_document_schema(pd.concat(
            [prepare_article_passages(articles), comments], ignore_index=True, sort=False,
        ))
    elif args.include_comments_in_fit:
        training_documents = _ensure_document_schema(
            pd.concat(
                [prepare_article_passages(articles), prepare_documents(_empty_articles(), comments)],
                ignore_index=True,
            )
        )
    else:
        training_documents = _ensure_document_schema(prepare_article_passages(articles))
    if args.fit_comment_documents and args.fit_role == "paper2_test":
        if len(comments) != len(prepare_article_passages(articles)):
            raise ValueError("Paper 2 fit comments must equal the article passage count")
    training_documents = training_documents.sort_values("doc_id").reset_index(drop=True)
    print(
        f"Loaded training corpus: {training_documents['story_id'].nunique():,} stories, "
        f"{len(articles):,} articles, {len(comments):,} published comments, "
        f"{len(training_documents):,} documents "
        f"({'article/comment' if (args.include_comments_in_fit or args.fit_comment_documents) else 'article-only'} fit).",
        flush=True,
    )
    fit_comment_source = (
        {"path": str(args.fit_comment_documents.resolve()),
         "sha256": _file_sha256(args.fit_comment_documents)} if args.fit_comment_documents else None
    )
    input_signature = {
        "fit_comment_documents": fit_comment_source,
        "split_sha256": _file_sha256(split_path),
        "analysis_comments_sha256": _file_sha256(analysis_comments_path),
        "data_root": str(data_root),
        "embedding_store": str(embedding_store),
        "year": int(args.year),
        "fit_scope": "all_stories" if args.all_stories else args.fit_role,
        "include_comments_in_fit": bool(args.include_comments_in_fit or args.fit_comment_documents),
        "calculate_probabilities": bool(args.calculate_probabilities),
        "configuration": json.loads(json.dumps(asdict(model_config))),
        "data_inventory": file_inventory(data_root),
        "embedding_inventory": file_inventory(embedding_store),
        "embedding_manifest_sha256": _file_sha256(embedding_store / "embedding_manifest.json"),
    }
    if args.runs_root:
        setup = dict(configuration=asdict(model_config), inputs=input_signature,
                     batch_size=args.batch_size, workers=args.workers, run_tag=args.run_tag,
                     code_hashes={name: _file_sha256(repo_root / name) for name in (
                         "commentgap_analysis/topic_modeling.py", "commentgap_analysis/embeddings.py",
                         "commentgap_analysis/topic_runs.py", "scripts/run_topic_model_fit.py")},
                     package_versions=_package_versions(), python=python_version())
        output_root, registered_run = prepare_run(args.runs_root.resolve(), setup)
        print(f"Registered run: {output_root.name}", flush=True)
        if registered_run["status"] == "completed":
            resolve_run(args.runs_root.resolve(), output_root.name)
            print(f"Using completed run: {output_root}", flush=True)
            return 0
    output_root.mkdir(parents=True, exist_ok=True)
    existing_progress = sorted(
        output_root.glob("membership_batches_*/progress.json"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    resumable_progress = next(
        (path for path in existing_progress if not json.loads(path.read_text()).get("completed", False)),
        None,
    )
    model_path = output_root / "topic_model.joblib"
    manifest_path = output_root / "topic_model_manifest.json"
    if model_path.exists() and manifest_path.exists() and (args.runs_root or resumable_progress is not None):
        if resumable_progress is not None:
            previous = json.loads(resumable_progress.read_text())
            if previous.get("input_signature") != input_signature or previous.get("batch_size") != args.batch_size:
                raise ValueError("Resume inputs/configuration/batch size changed; use a separate run directory")
        saved_config = json.loads(manifest_path.read_text())["config"]
        if saved_config != json.loads(json.dumps(asdict(model_config))):
            raise ValueError("Saved model configuration differs from the requested setup")
        bundle = load_topic_model(model_path)
        paths = {"model": model_path, "manifest": manifest_path}
        print(f"Resuming saved model in {output_root}.", flush=True)
    else:
        training_embeddings = load_precomputed_embeddings(training_documents, embedding_store)
        print("Loaded training BGE-M3 embeddings; fitting automated BERTopic.", flush=True)
        bundle = fit_topic_model(
            training_documents,
            model_config,
            embeddings=training_embeddings,
        )
        paths = save_topic_model(bundle, output_root)

    manifest_signature = hashlib.sha256(paths["manifest"].read_bytes()).hexdigest()[:12]
    batch_root = output_root / f"membership_batches_{manifest_signature}_paper2_test"
    batch_root.mkdir(parents=True, exist_ok=True)
    progress_path = batch_root / "progress.json"
    progress = {
        "model_manifest": str(paths["manifest"]),
        "manifest_signature": manifest_signature,
        "input_signature": input_signature,
        "batch_size": args.batch_size,
        "article_batches": [],
        "comment_batches": [],
        "article_done": 0,
        "comment_done": 0,
        "coverage": {},
    }
    if progress_path.exists():
        progress = json.loads(progress_path.read_text())
        if progress.get("manifest_signature") != manifest_signature:
            raise ValueError(f"Progress file belongs to a different model: {progress_path}")
        if progress.get("input_signature") != input_signature:
            raise ValueError(f"Progress file belongs to different inputs or settings: {progress_path}")
        if progress.get("batch_size") != args.batch_size:
            raise ValueError(f"Resume requires the original batch size ({progress.get('batch_size')}): {progress_path}")
    coverage = _coverage_from_json(progress.get("coverage", {}))
    article_total = 0
    comment_total = 0
    article_started = time.perf_counter()
    comment_started = time.perf_counter()

    try:
        con = duckdb.connect()
        article_total = con.execute(
            "SELECT COUNT(*) FROM read_parquet(?, hive_partitioning=true, union_by_name=true) "
            "WHERE year = ?" + test_filter,
            [str(embedding_store / "article_passages/**/*.parquet"), args.year, *test_params],
        ).fetchone()[0]
        comment_total = con.execute(
            "SELECT COUNT(*) FROM read_parquet(?, hive_partitioning=true, union_by_name=true) AS c "
            "INNER JOIN read_parquet(?) AS a "
            "ON CAST(c.story_id AS VARCHAR) = CAST(a.story_id AS VARCHAR) "
            "AND CAST(c.comment_id AS VARCHAR) = CAST(a.comment_id AS VARCHAR) "
            "WHERE c.year = ? AND lower(COALESCE(c.lifecycle_status, 'published')) = 'published'",
            [str(data_root / "comments/**/*.parquet"), str(analysis_comments_path), args.year],
        ).fetchone()[0]

        article_reader = con.execute(
            "SELECT CAST(story_id AS VARCHAR) AS story_id, title, subtitle, body "
            "FROM read_parquet(?, hive_partitioning=true, union_by_name=true) "
            "WHERE year = ?" + test_filter,
            [str(data_root / "articles/**/*.parquet"), args.year, *test_params],
        ).to_arrow_reader(args.batch_size)
        print(
            f"Starting article-passage transformation: {article_total:,} source passages "
            f"using {args.workers:,} workers and batches of at most {args.batch_size:,}.",
            flush=True,
        )
        _parallel_transform_batches(
            article_reader, bundle, embedding_store, batch_root, progress, coverage,
            "articles", article_total, article_started, args.batch_size, args.workers, prepare_article_passages,
            "article", "article passages",
        )

        comment_reader = con.execute(
            "SELECT CAST(c.story_id AS VARCHAR) AS story_id, CAST(c.comment_id AS VARCHAR) AS comment_id, c.effective_text, c.text "
            "FROM read_parquet(?, hive_partitioning=true, union_by_name=true) AS c "
            "INNER JOIN read_parquet(?) AS a "
            "ON CAST(c.story_id AS VARCHAR) = CAST(a.story_id AS VARCHAR) "
            "AND CAST(c.comment_id AS VARCHAR) = CAST(a.comment_id AS VARCHAR) "
            "WHERE c.year = ? AND lower(COALESCE(c.lifecycle_status, 'published')) = 'published' "
            "ORDER BY c.story_id, c.comment_id",
            [str(data_root / "comments/**/*.parquet"), str(analysis_comments_path), args.year],
        ).to_arrow_reader(args.batch_size)
        print(
            f"Starting comment transformation: {comment_total:,} source rows "
            f"using {args.workers:,} workers and batches of at most {args.batch_size:,}.",
            flush=True,
        )
        _parallel_transform_batches(
            comment_reader, bundle, embedding_store, batch_root, progress, coverage,
            "comments", comment_total, comment_started, args.batch_size, args.workers,
            lambda frame: prepare_documents(_empty_articles(), frame),
            "comment", "comments",
        )
    finally:
        if "con" in locals():
            con.close()

    membership_path = output_root / "document_topic_memberships.parquet"
    _merge_batch_files(batch_root, membership_path)
    memberships_coverage = _coverage_rows(coverage)
    memberships_coverage.to_parquet(output_root / "article_topic_coverage.parquet", index=False)
    memberships_coverage.to_csv(output_root / "article_topic_coverage.csv", index=False)
    memberships_coverage.loc[~memberships_coverage["article_has_valid_topic"]].to_csv(
        output_root / "articles_without_valid_topics.csv", index=False
    )
    memberships_coverage.loc[~memberships_coverage["has_valid_topic"]].to_csv(
        output_root / "stories_without_any_valid_topics.csv", index=False
    )
    quality = topic_model_quality_table(bundle)
    quality.to_csv(output_root / "topic_terms.csv", index=False)
    progress["completed"] = True
    progress["coverage"] = _coverage_for_json(coverage)
    _write_progress(progress_path, progress)
    manifest_data = json.loads(paths["manifest"].read_text())
    run_metadata = {
        "code_revision": _code_revision(repo_root),
        "code_hash": _file_sha256(repo_root / "commentgap_analysis/topic_modeling.py"),
        "platform": platform(),
        "python": python_version(),
        "package_versions": _package_versions(),
        "configuration": manifest_data.get("config"),
        "model_components": manifest_data.get("model_components"),
        "inputs": {
            "fit_comment_documents": fit_comment_source,
            "split": {"path": str(split_path), "sha256": _file_sha256(split_path)},
            "analysis_comments": {"path": str(analysis_comments_path), "sha256": _file_sha256(analysis_comments_path)},
            "embedding_store": str(embedding_store),
            "data_root": str(data_root),
            "fit_scope": "all_stories" if args.all_stories else args.fit_role,
            "include_comments_in_fit": bool(args.include_comments_in_fit or args.fit_comment_documents),
            "calculate_probabilities": bool(args.calculate_probabilities),
        },
        "counts": {
            "training_articles": int(len(articles)),
            "training_comments": int(len(comments)),
            "training_documents": int(len(training_documents)),
            "test_article_passages": int(article_total),
            "test_comments": int(comment_total),
            "output_stories": int(len(memberships_coverage)),
        },
        "model_manifest": {"path": str(paths["manifest"]), "sha256": _file_sha256(paths["manifest"])},
    }
    metadata_path = output_root / "topic_model_run_metadata.json"
    _write_progress(metadata_path, run_metadata)
    if json.loads(metadata_path.read_text()) != run_metadata:
        raise RuntimeError("Topic run metadata failed its JSON round trip")
    if args.runs_root:
        complete_run(output_root)
    outlier_count = int(memberships_coverage["n_outlier_sections"].sum())
    print(f"Final topics: {len(bundle.topic_terms)}", flush=True)
    print(f"Outlier documents: {outlier_count:,}", flush=True)
    print(f"Articles with no valid article topic: {int((~memberships_coverage['article_has_valid_topic']).sum()):,}", flush=True)
    print(f"Model: {paths['model']}", flush=True)
    print(f"Memberships: {membership_path}", flush=True)
    print(f"Coverage: {output_root / 'article_topic_coverage.csv'}", flush=True)
    print(f"Resume batches: {batch_root}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
