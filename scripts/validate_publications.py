#!/usr/bin/env python3
"""Verify frozen submission bytes and optionally compile into isolated outputs."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def verify(root: Path) -> list[dict]:
    with (root / "provenance/paper-assets.csv").open(newline="") as stream:
        assets = list(csv.DictReader(stream))
    failures = []
    for asset in assets:
        path = root / asset["submitted_path"]
        if not path.is_file():
            failures.append({"path": asset["submitted_path"], "error": "missing"})
        elif digest(path) != asset["submitted_sha256"]:
            failures.append({"path": asset["submitted_path"], "error": "checksum mismatch"})
    return failures


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--compile", action="store_true", help="Compile both papers outside the source folders")
    parser.add_argument("--run-id", default="publication-validation")
    args = parser.parse_args()
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", args.run_id):
        parser.error("run ID must be a single safe directory name")
    root = Path(__file__).resolve().parents[1]
    failures = verify(root)
    if failures:
        print(json.dumps({"status": "failed", "failures": failures}, indent=2))
        return 1
    report = {"status": "passed", "submission_checksums": "passed", "builds": []}
    if args.compile:
        output = root / "outputs" / args.run_id / "rendered"
        for protected in ("data", "artifacts/canonical", "artifacts/archive", "artifacts/frozen", "submissions", "provenance", "model_output"):
            if output.resolve().is_relative_to((root / protected).resolve()):
                raise ValueError(f"Build output cannot alias protected inputs: {protected}")
        if output.exists():
            raise FileExistsError(f"Use a new --run-id; build directory already exists: {output}")
        output.mkdir(parents=True)
        for paper in ("CG1", "CG2"):
            target = output / paper
            target.mkdir()
            source = target / "source"
            source.mkdir()
            with (root / "provenance/paper-assets.csv").open(newline="") as stream:
                for asset in csv.DictReader(stream):
                    if asset["paper"] != paper:
                        continue
                    destination = source / asset["asset"]
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copyfile(root / asset["submitted_path"], destination)
            command = ["latexmk", "-pdf", "-interaction=nonstopmode", "-halt-on-error", f"-outdir={target}", "main.tex"]
            with (target / "build-output.txt").open("w") as stream:
                result = subprocess.run(command, cwd=source, stdout=stream, stderr=subprocess.STDOUT)
            log = (target / "main.log").read_text(errors="replace") if (target / "main.log").exists() else ""
            unresolved = bool(re.search(r"(?:Reference|Citation).*undefined|There were undefined references", log))
            pages = None
            if (target / "main.pdf").exists():
                info = subprocess.run(["pdfinfo", str(target / "main.pdf")], capture_output=True, text=True, check=True).stdout
                match = re.search(r"^Pages:\s+(\d+)", info, flags=re.MULTILINE)
                pages = int(match.group(1)) if match else None
            passed = result.returncode == 0 and pages == 18 and not unresolved
            report["builds"].append({"paper": paper, "passed": passed, "pages": pages, "undefined_references": unresolved, "returncode": result.returncode})
        report["post_build_checksum_failures"] = verify(root)
        if report["post_build_checksum_failures"] or not all(item["passed"] for item in report["builds"]):
            report["status"] = "failed"
        (output / "validation.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
