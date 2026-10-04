"""Public replay must stay complete, small, and separate from private inputs."""

from __future__ import annotations

import ast
import json
from pathlib import Path
import pandas as pd

from commentgap_analysis.public_replay import verify_public_stage
from commentgap_analysis.paths import ProjectPaths


ROOT = Path(__file__).resolve().parents[1]
PUBLIC = ROOT / "artifacts/canonical"


def test_every_notebook_has_verified_public_products():
    manifest = json.loads((PUBLIC / "replay_manifest.json").read_text())
    assert set(manifest["stages"]) == {f"{stage:02d}" for stage in range(1, 17)}
    assert sum(record["bytes"] for record in manifest["products"].values()) < 50_000_000
    assert not any(relative.endswith(".png") for relative in manifest["products"])
    for stage in manifest["stages"]:
        verify_public_stage(stage, root=PUBLIC)


def test_public_manifest_has_no_comment_or_story_ids():
    manifest = json.loads((PUBLIC / "replay_manifest.json").read_text())
    forbidden = {"story_id", "comment_id", "title", "canonical_url", "text",
                 "body", "representative_docs", "author_id"}
    for relative, record in manifest["products"].items():
        assert not forbidden.intersection(name.lower() for name in record.get("columns", [])), relative
        assert (PUBLIC / relative).stat().st_size < 50 * 1024 * 1024


def test_replay_notebook_cells_use_named_data(monkeypatch):
    import matplotlib.pyplot as plt
    import IPython.display

    monkeypatch.setenv("COMMENTGAP_MODE", "replay")
    monkeypatch.chdir(ROOT)
    monkeypatch.setattr(IPython.display, "display", lambda *args, **kwargs: None)
    expected_variables = {
        "01": {"overview", "monthly", "by_section", "status"},
        "04": {"distribution_summary", "invariants", "within_story", "correlations", "status"},
        "05": {"sample_flow", "topic_summary", "feature_summary", "primary_topics"},
        "06": {"gap_summary", "gap_topics", "article_scores", "primary_topics"},
        "08": {"experiment", "completion", "winners", "ranking"},
        "09": {"associations", "shap_summary", "winners", "held_out_performance"},
        "10": {"summary", "policy_row_counts", "policy_summary", "marginal_effects"},
        "11": {"summary", "effects", "agreement"},
        "12": {"ranking_metrics", "regression_forum", "ml_shap_forum"},
        "13": {"feature_means", "matrices", "umap_cluster_membership"},
        "14": {"search", "frontier", "final_topic_summary", "split_role_counts", "analysis_comment_rows"},
        "15": {"baseline_analysis", "rarefaction_analysis", "vote_attention_curve", "oracle_summaries", "attention_sensitivity"},
        "16": {"coverage_summary", "concentration_effects", "combined_effects", "scatter_summary", "attention_sensitivity"},
    }
    for stage in (f"{number:02d}" for number in range(1, 17) if number != 7):
        notebook_path = next(ROOT.glob(f"{stage}_*.ipynb"))
        notebook = json.loads(notebook_path.read_text())
        namespace = {"__name__": "__main__"}
        sources = ["".join(cell["source"]) for cell in notebook["cells"] if cell["cell_type"] == "code"]
        assert any(f"verify_public_stage('{stage}', root=PUBLIC_ROOT)" in source for source in sources)
        assert any("pd.read_csv((PUBLIC_ROOT /" in source or "pd.read_parquet((PUBLIC_ROOT /" in source
                   or "pd.read_csv(PUBLIC_ROOT /" in source or "pd.read_parquet(PUBLIC_ROOT /" in source
                   or "json.loads((PUBLIC_ROOT /" in source
                   for source in sources)
        assert all("display_replay_cell" not in source and "PUBLIC_REPLAY_STATE" not in source
                   and "load_replay_stage" not in source and "replay_table(" not in source
                   and "replay_item(" not in source and "replay_root(" not in source
                   for source in sources)
        for index, cell in enumerate(notebook["cells"]):
            if cell["cell_type"] == "code":
                source = "".join(cell["source"])
                exec(compile(source, f"{notebook_path.name}:cell{index}", "exec"), namespace)
        assert namespace["PUBLIC_ROOT"] == PUBLIC
        assert expected_variables.get(stage, set()) <= namespace.keys(), stage
        plt.close("all")


def test_attention_sensitivity_is_displayed_in_replay():
    relative = ("CG2/topics/runs/mcs15_nn10_ms5_seed2026_1e1b8fce5deb3e58/"
                "analysis/topic_policy_fitted_attention_sensitivity.csv")
    expected = pd.read_csv(PUBLIC / relative)
    assert expected.shape == (180, 10)
    for notebook_name, cell_index in (("15_topic_agenda_calculations.ipynb", 11),
                                      ("16_topic_agenda_analysis.ipynb", 6)):
        notebook = json.loads((ROOT / notebook_name).read_text())
        displayed = []
        namespace = {"mode": "replay", "PUBLIC_ROOT": PUBLIC,
                     "pd": pd, "display": displayed.append}
        exec("".join(notebook["cells"][cell_index]["source"]), namespace)
        pd.testing.assert_frame_equal(namespace["attention_sensitivity"], expected)
        assert len(displayed) == 1
        pd.testing.assert_frame_equal(displayed[0], expected.round(4))


def test_forum_input_counts_are_displayed_without_private_rows():
    notebook = json.loads((ROOT / "10_calculate_forum_scores.ipynb").read_text())
    displayed = []
    namespace = {"mode": "replay", "PUBLIC_ROOT": PUBLIC,
                 "pd": pd, "display": lambda *values: displayed.extend(values)}
    exec("".join(notebook["cells"][10]["source"]), namespace)
    summary = namespace["summary"]
    counts = namespace["policy_row_counts"]
    assert summary.set_index("quantity").loc["score rows", "value"] == counts["rows"].sum()
    assert list(counts) == ["ordering", "reply_mode", "pinned", "rows"]
    pd.testing.assert_frame_equal(displayed[0], summary)
    pd.testing.assert_frame_equal(
        displayed[1], counts.set_index(["ordering", "reply_mode", "pinned"])
    )


def test_notebooks_have_no_empty_or_opaque_replay_branches():
    forbidden = ("PUBLIC_REPLAY_WRAPPER_V1", "load_replay_stage", "replay_table(",
                 "replay_item(", "replay_root(", "replay_products", "replay_mode()",
                 "_COMMENTGAP_FROZEN_REPLAY")
    for notebook_path in ROOT.glob("[0-9][0-9]_*.ipynb"):
        notebook = json.loads(notebook_path.read_text())
        setup = "".join(notebook["cells"][0]["source"])
        assert setup.count('mode = "replay"') == 1, notebook_path.name
        assert "execution_mode()" not in setup, notebook_path.name
        for index, cell in enumerate(notebook["cells"]):
            if cell["cell_type"] != "code":
                continue
            source = "".join(cell["source"])
            assert not any(name in source for name in forbidden), (notebook_path.name, index)
            if index:
                assert not any(isinstance(node, (ast.Import, ast.ImportFrom))
                               for node in ast.walk(ast.parse(source))), (notebook_path.name, index)
            for node in ast.walk(ast.parse(source)):
                if isinstance(node, ast.If) and ast.unparse(node.test) == "mode == 'replay'":
                    assert not (len(node.body) == 1 and isinstance(node.body[0], ast.Pass)), (
                        notebook_path.name, index)
                    replay_calls = (child for statement in node.body for child in ast.walk(statement)
                                    if isinstance(child, ast.Call) and isinstance(child.func, ast.Name))
                    assert not any(call.func.id in {"display", "print"}
                                   or call.func.id.startswith("plot_") for call in replay_calls), (
                        notebook_path.name, index)


def test_public_plot_points_have_no_source_ids():
    points = pd.read_parquet(PUBLIC / "CG1/comment_gap/article_plot_coordinates.parquet")
    assert set(points) == {"scope", "n_candidates", "n_picks", "gap_score"}
    assert points["gap_score"].between(0, 1).all()


def test_private_and_public_paths_are_distinct(tmp_path, monkeypatch):
    entries = {"frozen_cg1_reporting": "artifacts/canonical/CG1/reporting"}
    public = tmp_path / entries["frozen_cg1_reporting"]
    private = tmp_path / "artifacts/archive/recompute/CG1/reporting"
    public.mkdir(parents=True)
    private.mkdir(parents=True)
    paths = ProjectPaths(root=tmp_path, entries=entries)
    monkeypatch.setenv("COMMENTGAP_MODE", "replay")
    assert paths.read_root("frozen_cg1_reporting") == public
    monkeypatch.setenv("COMMENTGAP_MODE", "recompute")
    assert paths.read_root("frozen_cg1_reporting") == private
