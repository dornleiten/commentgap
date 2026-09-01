from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import unittest

from unittest.mock import patch
import numpy as np
import pandas as pd

import commentgap_analysis.neural_ranking as neural_ranking

from commentgap_analysis.neural_ranking import (
    FeatureScaler,
    FrozenFusionRanker,
    MetadataOnlyRanker,
    ProgressReporter,
    _load_model_checkpoint,
    _load_workflow_state,
    _seed_everything,
    _selection_plan,
    _save_training_checkpoint,
    _save_workflow_state,
    _train_epoch,
    _validation_macro_ndcg,
    apply_fold_feature_columns,
    default_recipe,
    discover_complete_embedding_store,
    evaluate_tie_draws,
    fit_feature_scaler,
    lambda_ndcg_pair_weights,
    sample_pair_indices,
    resolve_model_revision,
    scores_to_long,
)
from commentgap_analysis.neural_ranker_cli import build_parser


class NeuralRankingTests(unittest.TestCase):
    def test_training_mode_cli_choices(self):
        parser = build_parser("metadata_mlp")
        self.assertEqual(
            parser.parse_args(["--training-mode", "fixed_split"]).training_mode,
            "fixed_split",
        )
        self.assertEqual(
            parser.parse_args(["--training-mode", "full"]).training_mode,
            "full",
        )

    def test_metadata_recipe_is_tabular_only(self):
        recipe = default_recipe("metadata_mlp")
        self.assertEqual(recipe.approach, "metadata_mlp")
        self.assertEqual(recipe.max_epochs, 30)
        self.assertEqual(recipe.patience, 3)
        self.assertEqual(recipe.pair_batch_size, 1024)
        self.assertEqual(recipe.ranking_loss, "lambda_ndcg")
        self.assertEqual(recipe.hard_negatives_per_positive, 1)
        self.assertEqual(recipe.gradient_accumulation, 1)

    def test_full_mode_has_no_selection_fold(self):
        self.assertEqual(_selection_plan("cv"), ((0, 1, 2, 3, 4), 5, "CV"))
        self.assertEqual(_selection_plan("fixed_split"), ((0,), 1, "fixed split"))
        self.assertEqual(_selection_plan("full"), ((), 0, "full training"))

    def test_model_initialization_is_reproducible(self):
        import torch

        recipe = default_recipe("metadata_mlp")
        _seed_everything(17)
        first = MetadataOnlyRanker(3, recipe)
        first_state = {name: value.detach().clone() for name, value in first.state_dict().items()}
        _seed_everything(17)
        second = MetadataOnlyRanker(3, recipe)
        for name, value in second.state_dict().items():
            torch.testing.assert_close(value, first_state[name])

    def test_all_audience_draws_are_averaged_once(self):
        import torch

        class TwoSelectorLinear(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.weights = torch.nn.Parameter(torch.zeros(2))

            def score(self, payload, metadata, selector):
                del payload
                return self.weights[selector] * metadata[:, 0]

        frame = pd.DataFrame(
            {
                "story_id": ["s", "s"],
                "comment_id": ["positive", "negative"],
                "x": [1.0, 0.0],
                "curator_selected": [1, 0],
                **{
                    f"audience_selected_draw_{draw:02d}": [1, 0]
                    for draw in range(1, 11)
                },
            }
        )
        scaler = FeatureScaler(
            features=("x",), means=(0.0,), scales=(1.0,), standardized=(False,)
        )
        recipe = neural_ranking.NeuralTrainingRecipe(
            **{
                **neural_ranking.asdict(default_recipe("metadata_mlp")),
                "negatives_per_positive": 1,
            }
        )
        model = TwoSelectorLinear()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        with patch.object(
            neural_ranking, "_autocast", wraps=neural_ranking._autocast
        ) as autocast:
            _train_epoch(
                model,
                frame,
                ["s"],
                None,
                scaler,
                optimizer,
                recipe,
                epoch=1,
                device="cpu",
            )
        self.assertAlmostEqual(float(model.weights[0].detach()), 0.05, places=6)
        self.assertAlmostEqual(float(model.weights[1].detach()), 0.05, places=6)
        self.assertTrue(all(call.args[1] is False for call in autocast.call_args_list))

    def test_validation_metric_averages_all_audience_draws(self):
        scored = pd.DataFrame(
            {
                "story_id": ["s", "s", "s"],
                "comment_id": ["a", "b", "c"],
                "n_picks": [1, 1, 1],
                "curator_selected": [1, 0, 0],
                "audience_score": [3.0, 2.0, 1.0],
                "curator_score": [3.0, 2.0, 1.0],
                "audience_selected_draw_01": [1, 0, 0],
                **{
                    f"audience_selected_draw_{draw:02d}": [0, 0, 1]
                    for draw in range(2, 11)
                },
            }
        )
        self.assertAlmostEqual(_validation_macro_ndcg(scored), 0.55, places=6)

    def test_workflow_rejects_unknown_training_mode(self):
        with self.assertRaisesRegex(ValueError, "training_mode"):
            neural_ranking.run_neural_ranker_workflow(
                Path("unused-model-data"),
                Path("unused-data"),
                Path("unused-output"),
                approach="frozen_bge",
                training_mode="not-a-mode",
            )

    def test_progress_reporter_prints_eta_and_candidate_counts(self):
        output = io.StringIO()
        with patch.object(
            neural_ranking.time,
            "monotonic",
            side_effect=[0.0, 2.0, 4.0],
        ):
            with redirect_stdout(output):
                progress = ProgressReporter("test inference", 2, every=10)
                progress.update(candidates=3)
                progress.update(candidates=5)
        rendered = output.getvalue()
        self.assertIn("0/2 articles", rendered)
        self.assertIn("1/2 articles (50.0%)", rendered)
        self.assertIn("ETA=00:00:02", rendered)
        self.assertIn("2/2 articles (100.0%) candidates=8", rendered)

    def test_scaler_uses_registry_and_training_values_only(self):
        training = pd.DataFrame({"continuous": [1.0, 3.0], "binary": [0, 1]})
        scaler = fit_feature_scaler(
            training,
            ["continuous", "binary"],
            {
                "continuous": {"standardize": True},
                "binary": {"standardize": False},
            },
        )
        transformed = scaler.transform(
            pd.DataFrame({"continuous": [5.0], "binary": [1]})
        )
        self.assertAlmostEqual(transformed[0, 0], 3 / np.sqrt(2), places=6)
        self.assertEqual(transformed[0, 1], 1.0)

    def test_fold_replacement_does_not_change_source_frame(self):
        frame = pd.DataFrame(
            {"reply_depth_centered": [10.0], "reply_depth_centered_fold_02": [-1.0]}
        )
        result = apply_fold_feature_columns(
            frame,
            ["reply_depth_centered"],
            {"reply_depth_centered": "reply_depth_centered_fold_02"},
        )
        self.assertEqual(result["reply_depth_centered"].iloc[0], -1.0)
        self.assertEqual(frame["reply_depth_centered"].iloc[0], 10.0)

    def test_pair_sampler_is_deterministic_and_query_local(self):
        labels = np.array([1, 0, 0, 1, 0])
        first = sample_pair_indices(labels, negatives_per_positive=3, seed=7)
        second = sample_pair_indices(labels, negatives_per_positive=3, seed=7)
        np.testing.assert_array_equal(first, second)
        self.assertEqual(first.shape, (6, 2))
        self.assertTrue((labels[first[:, 0]] == 1).all())
        self.assertTrue((labels[first[:, 1]] == 0).all())

    def test_pair_sampler_includes_highest_scoring_negative(self):
        labels = np.array([1, 0, 0, 1, 0])
        scores = np.array([0.0, 0.2, 2.0, 0.1, 1.0])
        pairs = sample_pair_indices(
            labels,
            negatives_per_positive=2,
            scores=scores,
            hard_negatives_per_positive=1,
            seed=7,
        )
        for positive in np.flatnonzero(labels == 1):
            negative_rows = pairs[pairs[:, 0] == positive, 1]
            self.assertIn(2, negative_rows)

    def test_lambda_ndcg_weights_use_current_top_k_swap_cost(self):
        labels = np.array([1, 0, 0])
        scores = np.array([0.0, 3.0, 2.0])
        pairs = np.array([[0, 1], [0, 2]])
        weights = lambda_ndcg_pair_weights(labels, scores, pairs, k=1)
        np.testing.assert_allclose(weights, np.array([1.0, 0.0]))


    def test_scores_export_both_selectors_and_all_audience_draws(self):
        scored = pd.DataFrame(
            {
                "story_id": ["s", "s", "s"],
                "comment_id": ["a", "b", "c"],
                "n_picks": [1, 1, 1],
                "curator_selected": [1, 0, 0],
                "audience_score": [0.1, 0.8, 0.2],
                "curator_score": [0.9, 0.1, 0.2],
                **{
                    f"audience_selected_draw_{draw:02d}": [0, 1, 0]
                    for draw in range(1, 11)
                },
            }
        )
        long = scores_to_long(scored)
        self.assertEqual(set(long["selector"]), {"audience", "curator"})
        self.assertEqual(len(long), 6)
        ties = evaluate_tie_draws(scored)
        self.assertEqual(
            set(ties.loc[ties["selector"] == "audience", "audience_tie_draw"]),
            set(range(1, 11)),
        )

    def test_exact_revision_does_not_require_hub_resolution(self):
        revision = "A" * 40
        self.assertEqual(resolve_model_revision("unused/model", revision), revision.lower())

    def test_metadata_ranker_scores_both_selectors(self):
        import torch

        model = MetadataOnlyRanker(3, default_recipe("metadata_mlp"))
        scores = model.score_both(None, torch.zeros((4, 3)))
        self.assertEqual(tuple(scores.shape), (4, 2))
    def test_embedding_resolver_rejects_subset_and_selects_complete(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name, watermark in (("subset", "SUBSET"), ("full", "COMPLETE_SOURCE")):
                build = root / "model=bge" / f"build={name}"
                build.mkdir(parents=True)
                (build / "embedding_manifest.json").write_text(
                    json.dumps(
                        {
                            "status": "complete",
                            "watermark": watermark,
                            "embedding_dimension": 1024,
                            "model": {
                                "model_id": "BAAI/bge-m3",
                                "resolved_revision": "rev",
                            },
                        }
                    )
                )
            store, manifest = discover_complete_embedding_store(
                root, model_id="BAAI/bge-m3", revision="rev"
            )
            self.assertEqual(store.name, "build=full")
            self.assertEqual(manifest["watermark"], "COMPLETE_SOURCE")

    def test_frozen_ranker_can_learn_pair_direction(self):
        import torch

        torch.manual_seed(2)
        recipe = default_recipe("frozen_bge")
        model = FrozenFusionRanker(2, recipe)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        embeddings = torch.zeros((2, 1024))
        embeddings[0, 0] = 1
        embeddings[1, 0] = -1
        metadata = torch.zeros((2, 2))
        selector = torch.zeros(2, dtype=torch.long)
        for _ in range(30):
            scores = model.score(embeddings, metadata, selector)
            loss = torch.nn.functional.softplus(-(scores[0] - scores[1]))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        final = model.score(embeddings, metadata, selector)
        self.assertGreater(float(final[0].detach()), float(final[1].detach()))


    def test_training_checkpoint_round_trip(self):
        import torch

        torch.manual_seed(3)
        recipe = default_recipe("frozen_bge")
        model = FrozenFusionRanker(2, recipe)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        scaler = fit_feature_scaler(
            pd.DataFrame({"x": [0.0, 1.0]}),
            ["x"],
            {"x": {"standardize": True}},
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "fold_00_resume.pt"
            _save_training_checkpoint(
                model,
                optimizer,
                path,
                approach="frozen_bge",
                workflow_signature="sig",
                scope="root",
                phase="cv",
                epoch=1,
                best_metric=0.5,
                best_epoch=1,
                stale=0,
                scaler=scaler,
                fold=0,
            )
            restored = FrozenFusionRanker(2, recipe)
            restored_optimizer = torch.optim.Adam(restored.parameters(), lr=0.01)
            payload = _load_model_checkpoint(restored, path, "frozen_bge")
            restored_optimizer.load_state_dict(payload["optimizer"])
            self.assertEqual(payload["checkpoint_kind"], "training_resume")
            self.assertEqual(payload["epoch"], 1)
            for name, parameter in model.state_dict().items():
                torch.testing.assert_close(parameter, restored.state_dict()[name])

    def test_workflow_state_round_trip_rejects_wrong_signature(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "workflow_state.json"
            _save_workflow_state(
                path,
                workflow_signature="sig",
                scope="root",
                phase="cv",
                completed_folds=[0, 1],
                selected_epochs=[1, 2],
                cv_rows=[{"fold": 0, "epoch": 1}],
            )
            loaded = _load_workflow_state(
                path, workflow_signature="sig", scope="root", resume=True
            )
            self.assertEqual(loaded["phase"], "cv")
            self.assertEqual(loaded["completed_folds"], [0, 1])
            self.assertEqual(
                _load_workflow_state(
                    path, workflow_signature="different", scope="root", resume=True
                ),
                {},
            )


if __name__ == "__main__":
    unittest.main()
