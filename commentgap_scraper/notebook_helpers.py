"""Reusable helpers for the scraping and collection-summary notebook."""

from __future__ import annotations

import os
import shlex
import subprocess
import sys
from collections.abc import Iterable, Mapping
from pathlib import Path


def run_scraper(
    *arguments: object,
    project_root: Path,
    year: int,
    output_dir: Path,
    python_executable: str | Path = sys.executable,
    allow_failure: bool = False,
) -> int:
    """Run the repository scraper with the notebook's year and output directory."""
    command = [
        str(python_executable),
        "-m",
        "commentgap_scraper",
        *map(str, arguments),
        "--year",
        str(year),
        "--output",
        str(output_dir),
    ]
    print("$ " + " ".join(shlex.quote(item) for item in command))
    result = subprocess.run(command, cwd=project_root, env=os.environ.copy())
    if result.returncode and not allow_failure:
        raise subprocess.CalledProcessError(result.returncode, command)
    if result.returncode:
        print(f"Scraper exited with status {result.returncode}; inspect validation output below.")
    return result.returncode


def require_collection_credentials(
    contact: str | None = None,
    *,
    include_hash_key: bool = False,
    environ: Mapping[str, str] | None = None,
) -> None:
    """Require the environment values needed by collection actions."""
    environment = os.environ if environ is None else environ
    missing = []
    if not contact:
        missing.append("COMMENTGAP_CONTACT")
    if include_hash_key and not environment.get("COMMENTGAP_HASH_KEY"):
        missing.append("COMMENTGAP_HASH_KEY")
    if missing:
        raise RuntimeError("Set these environment variables before collecting: " + ", ".join(missing))


def parquet_files(output_dir: Path, table_name: str) -> list[Path]:
    """Return month-partitioned Parquet files for a scraper table."""
    return sorted((output_dir / table_name).glob("year=*/month=*/*.parquet"))


def sql_string_list(paths: Iterable[Path]) -> str:
    """Render paths as a SQL list of safely escaped string literals."""
    quoted = ["'" + str(path).replace("'", "''") + "'" for path in paths]
    return "[" + ", ".join(quoted) + "]"


def parquet_relation(output_dir: Path, table_name: str) -> str | None:
    """Return a DuckDB relation expression for an available scraper table."""
    paths = parquet_files(output_dir, table_name)
    if not paths:
        return None
    return f"read_parquet({sql_string_list(paths)}, hive_partitioning=true, union_by_name=true)"
