"""Configuration-addressed topic runs and explicit downstream selection."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')
    temporary.replace(path)


def setup_id(setup: dict) -> str:
    return hashlib.sha256(json.dumps(setup, sort_keys=True).encode()).hexdigest()[:16]


def file_inventory(root: Path, pattern: str = '**/*.parquet') -> list[dict]:
    """Detect changed shards cheaply; size/mtime are not content checksums."""
    return [dict(path=str(p.relative_to(root)), size=p.stat().st_size,
                 mtime_ns=p.stat().st_mtime_ns) for p in sorted(root.glob(pattern))]


def prepare_run(runs_root: Path, setup: dict) -> tuple[Path, dict]:
    config = setup['configuration']
    name = (f"mcs{config['hdbscan_min_cluster_size']}_nn{config['umap_n_neighbors']}"
            f"_ms{config['hdbscan_min_samples']}_seed{config['random_state']}"
            f"_{setup_id(setup)}")
    root = runs_root / name
    manifest_path = root / 'run.json'
    # Normalize tuples and other JSON-compatible containers before comparison.
    setup = json.loads(json.dumps(setup))
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        if manifest['setup'] != setup:
            raise ValueError(f'Run identity collision: {root}')
    else:
        if root.exists() and any(root.iterdir()):
            raise ValueError(f'Refusing unregistered nonempty run directory: {root}')
        manifest = dict(run_id=name, status='running', setup=setup,
                        created_at=datetime.now(timezone.utc).isoformat())
        write_json(manifest_path, manifest)
    return root, manifest


def complete_run(root: Path) -> None:
    manifest = json.loads((root / 'run.json').read_text())
    manifest.update(status='completed', completed_at=datetime.now(timezone.utc).isoformat())
    write_json(root / 'run.json', manifest)


def list_runs(runs_root: Path) -> list[dict]:
    rows = []
    for path in sorted(runs_root.glob('*/run.json')):
        manifest = json.loads(path.read_text())
        rows.append(dict(run_id=manifest['run_id'], status=manifest['status'],
                         created_at=manifest['created_at'],
                         **manifest['setup']['configuration']))
    return rows


def resolve_run(runs_root: Path, run_id: str | None = None) -> Path:
    """Use an explicit ID, or the shared selection; never guess the latest run."""
    if run_id is None:
        selection = runs_root / 'selected_run.json'
        if not selection.exists():
            raise FileNotFoundError('Select a completed topic run in notebook 14 first.')
        run_id = json.loads(selection.read_text())['run_id']
    if not run_id or Path(run_id).name != run_id or run_id in {'.', '..'}:
        raise ValueError('run_id must be a directory name within runs_root')
    root = runs_root / run_id
    manifest = json.loads((root / 'run.json').read_text())
    if manifest['status'] != 'completed':
        raise ValueError(f'Topic run is not completed: {run_id}')
    for name in ('topic_model.joblib', 'topic_model_manifest.json',
                 'topic_model_run_metadata.json', 'document_topic_memberships.parquet',
                 'article_topic_coverage.parquet'):
        if not (root / name).exists():
            raise FileNotFoundError(root / name)
    return root


def select_run(runs_root: Path, run_id: str) -> Path:
    root = resolve_run(runs_root, run_id)
    write_json(runs_root / 'selected_run.json', dict(run_id=run_id))
    return root
