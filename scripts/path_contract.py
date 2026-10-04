#!/usr/bin/env python3
"""Small JSON bridge from R entry points to the shared Python path contract."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from commentgap_analysis.paths import ExecutionContext  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--request", type=Path, required=True)
    args = parser.parse_args()
    request = json.loads(args.request.read_text(encoding="utf-8"))
    context = ExecutionContext.from_values(
        mode=request.get("mode"),
        run_id=request.get("run_id"),
        repo_root=request.get("repo_root"),
        cwd=request.get("cwd"),
    )
    for value in request.get("inputs", []):
        context.read_path(value)
    command = request.get("command")
    if command == "read_root":
        result = context.read_root(
            request["key"],
            explicit=request.get("explicit"),
            env_var=request.get("env_var"),
        )
    elif command == "prepare":
        result = context.prepare_run(contract=request.get("contract") or {})
    elif command == "stage":
        if request.get("kind", "output") == "staging":
            result = context.staging_output(
                request["area"], explicit=request.get("explicit")
            )
        else:
            result = context.output_root(
                request["area"],
                explicit=request.get("explicit"),
                contract=request.get("contract") or {},
            )
    elif command == "stage_path":
        # Validate an already prepared stage destination without attempting
        # to create or re-register the stage.  R Markdown chunks use this
        # when the render wrapper has prepared the manifest first.
        expected = context.run_path(request["area"])
        explicit = request.get("explicit")
        candidate = expected if explicit is None else Path(explicit)
        if not candidate.is_absolute():
            candidate = Path(request.get("cwd") or Path.cwd()) / candidate
        if candidate.resolve(strict=False) != expected.resolve(strict=False):
            raise ValueError(f"Stage output must remain under this run: {expected}")
        result = context.paths.require_writable(
            candidate, cwd=request.get("cwd"), additional_protected=tuple(context.read_inputs)
        )
    elif command == "stage_child":
        stage_root = context.paths.require_writable(request["stage_root"], cwd=request.get("cwd"))
        result = context.paths.require_writable(request["destination"], cwd=request.get("cwd"))
        if not result.is_relative_to(stage_root):
            raise ValueError(f"Writable R output must remain below its prepared stage: {result}")
    elif command == "guard":
        result = context.paths.require_writable(
            request["destination"], cwd=request.get("cwd")
        )
    else:
        raise ValueError(f"Unknown path-contract command: {command}")
    print(json.dumps({"path": str(result), "mode": context.mode, "run_id": context.run_id}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
