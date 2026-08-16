from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path

from . import __version__
from .config import ScrapeConfig
from .crawler import crawl, discover
from .legacy import export_legacy
from .migration import migrate_existing
from .validation import validate_dataset


def _default_output(year: int) -> Path:
    return Path("data") / f"scrape_{year}"


def _add_year_output(parser: argparse.ArgumentParser, *, allow_month: bool = False) -> None:
    parser.add_argument("--year", type=int, default=2025)
    if allow_month:
        parser.add_argument(
            "--month",
            type=int,
            choices=range(1, 13),
            metavar="1-12",
            help="Restrict the operation to one publication month",
        )
    parser.add_argument(
        "--output",
        type=Path,
        help="Output directory (default: data/scrape_YEAR)",
    )


def _add_network(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--user-agent",
        default=os.environ.get("COMMENTGAP_USER_AGENT"),
        help="Operator/project identifier; or set COMMENTGAP_USER_AGENT",
    )
    parser.add_argument(
        "--contact",
        default=os.environ.get("COMMENTGAP_CONTACT"),
        help="Operator contact email; or set COMMENTGAP_CONTACT",
    )
    parser.add_argument("--request-interval", type=float, default=1.0)
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument("--max-retries", type=int, default=5)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="commentgap-scrape",
        description="Collect a resumable, pseudonymized Der Standard forum dataset.",
    )
    parser.add_argument("--version", action="version", version=__version__)
    parser.add_argument(
        "--log-level",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
        default="INFO",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    discover_parser = subparsers.add_parser("discover", help="Discover stories from monthly sitemaps")
    _add_year_output(discover_parser, allow_month=True)
    _add_network(discover_parser)

    crawl_parser = subparsers.add_parser("crawl", help="Collect articles, forums, and comments")
    _add_year_output(crawl_parser, allow_month=True)
    _add_network(crawl_parser)
    crawl_parser.add_argument(
        "--hash-key-env",
        default="COMMENTGAP_HASH_KEY",
        help="Environment variable containing the HMAC key",
    )
    crawl_parser.add_argument(
        "--confirm-existing-hash-key",
        action="store_true",
        help=(
            "Override an inconclusive automatic legacy-key check by confirming that "
            "the supplied key is the original key used by the existing dataset"
        ),
    )
    crawl_parser.add_argument("--limit", type=int, help="Process at most this many stories")
    crawl_parser.add_argument(
        "--selection",
        choices=(
            "chronological",
            "monthly-random",
            "stratified-pilot",
            "monthly-round-robin",
        ),
        default="chronological",
        help=(
            "Ordering used with --limit; stratified-pilot is recommended for pilots "
            "(monthly-round-robin is a compatibility alias)"
        ),
    )
    crawl_parser.add_argument(
        "--selection-seed",
        type=int,
        default=2025,
        help="Reproducible seed used by randomized selection modes (default: 2025)",
    )
    crawl_parser.add_argument(
        "--pilot-candidate-pool",
        type=int,
        default=500,
        help=(
            "Candidates whose forum sizes are inspected by stratified-pilot "
            "before selecting --limit stories (default: 500)"
        ),
    )
    retry_group = crawl_parser.add_mutually_exclusive_group()
    retry_group.add_argument(
        "--retry-failed", action="store_true", help="Include previously failed stories"
    )
    retry_group.add_argument(
        "--only-failed",
        action="store_true",
        help=(
            "Process only failed or interrupted stories; excludes untouched pending stories"
        ),
    )
    crawl_parser.add_argument(
        "--story-id",
        action="append",
        dest="story_ids",
        help="Restrict to a story ID; may be repeated",
    )
    crawl_parser.add_argument(
        "--reply-query-depth",
        type=int,
        default=32,
        help="Maximum nested reply depth requested from GraphQL",
    )
    crawl_parser.add_argument(
        "--progress-every-pages",
        type=int,
        default=10,
        help="Report progress within a large forum every N GraphQL pages (default: 10)",
    )

    validate_parser = subparsers.add_parser("validate", help="Validate completeness and invariants")
    _add_year_output(validate_parser, allow_month=True)
    validate_parser.add_argument(
        "--allow-incomplete",
        action="store_true",
        help="Validate a pilot without requiring all discovered stories to be terminal",
    )

    legacy_parser = subparsers.add_parser(
        "export-legacy", help="Export normalized tables using the original column names"
    )
    _add_year_output(legacy_parser)
    migration_parser = subparsers.add_parser(
        "migrate-existing",
        help="Offline upgrade of existing Parquet files to the current schema",
    )
    _add_year_output(migration_parser)
    return parser


def _output(args: argparse.Namespace) -> Path:
    return (args.output or _default_output(args.year)).resolve()


def _config(args: argparse.Namespace) -> ScrapeConfig:
    if not args.user_agent:
        raise ValueError("--user-agent or COMMENTGAP_USER_AGENT is required")
    if not args.contact:
        raise ValueError("--contact or COMMENTGAP_CONTACT is required")
    return ScrapeConfig(
        year=args.year,
        output_dir=_output(args),
        user_agent=args.user_agent,
        contact=args.contact,
        month=getattr(args, "month", None),
        request_interval=args.request_interval,
        timeout=args.timeout,
        max_retries=args.max_retries,
        reply_query_depth=getattr(args, "reply_query_depth", 32),
        progress_every_pages=getattr(args, "progress_every_pages", 10),
    )


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    try:
        if args.command == "discover":
            result = discover(_config(args))
            print(json.dumps(result, indent=2, sort_keys=True))
            return 0
        if args.command == "crawl":
            key = os.environ.get(args.hash_key_env)
            if not key:
                raise ValueError(f"{args.hash_key_env} must be set to a secret of at least 16 bytes")
            result = crawl(
                _config(args),
                hash_key=key,
                limit=args.limit,
                retry_failed=args.retry_failed,
                only_failed=args.only_failed,
                story_ids=args.story_ids,
                monthly_random=args.selection in {"monthly-random", "monthly-round-robin"},
                selection_seed=args.selection_seed,
                stratified_pilot=args.selection == "stratified-pilot",
                pilot_candidate_pool=args.pilot_candidate_pool,
                confirm_existing_hash_key=args.confirm_existing_hash_key,
            )
            print(json.dumps(result, indent=2, sort_keys=True))
            return 0 if result.get("failed", 0) == 0 else 2
        if args.command == "validate":
            result = validate_dataset(
                _output(args),
                args.year,
                month=args.month,
                allow_incomplete=args.allow_incomplete,
            )
            print(json.dumps(result, indent=2, sort_keys=True))
            return 0 if result["passed"] else 2
        if args.command == "export-legacy":
            comments, articles = export_legacy(_output(args), args.year)
            print(json.dumps({"comments": str(comments), "articles": str(articles)}, indent=2))
            return 0
        if args.command == "migrate-existing":
            result = migrate_existing(_output(args), args.year)
            print(json.dumps(result, indent=2, sort_keys=True))
            return 0
    except (RuntimeError, ValueError) as exc:
        logging.getLogger("commentgap_scraper").error("%s", exc)
        return 2
    parser.error(f"unsupported command: {args.command}")
    return 2
