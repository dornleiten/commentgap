import unittest

from commentgap_analysis.presentation_labels import (
    ORDERING_LABELS,
    OUTCOME_DISPLAY_LABELS,
    OUTCOME_DISPLAY_ORDER,
    REPLY_DISPLAY_MARKERS,
    model_selector_label,
)


class PresentationLabelTests(unittest.TestCase):
    def test_final_model_ordering_labels(self):
        self.assertEqual(
            list(ORDERING_LABELS.values()),
            [
                "Reg: Editor",
                "Reg: Audience",
                "XGB: Editor",
                "XGB: Audience",
                "XGB-T: Editor",
                "XGB-T: Audience",
                "NN: Editor",
                "NN: Audience",
                "NN-T: Editor",
                "NN-T: Audience",
            ],
        )

    def test_selector_labels_do_not_change_internal_policy_ids(self):
        self.assertEqual(
            model_selector_label("neural", "metadata_bge", "curator"),
            "NN-T: Editor",
        )
        self.assertEqual(
            model_selector_label("xgboost", "metadata", "audience"),
            "XGB: Audience",
        )
        self.assertEqual(
            set(ORDERING_LABELS),
            {
                "regression_editor",
                "regression_audience",
                "xgb_metadata_editor",
                "xgb_metadata_audience",
                "xgb_metadata_text_editor",
                "xgb_metadata_text_audience",
                "neural_metadata_editor",
                "neural_metadata_audience",
                "neural_metadata_text_editor",
                "neural_metadata_text_audience",
            },
        )

    def test_outcome_display_order_and_labels(self):
        self.assertEqual(
            [OUTCOME_DISPLAY_LABELS[outcome] for outcome in OUTCOME_DISPLAY_ORDER],
            [
                "Comment-Article similarity",
                "Comment novelty",
                "Author incumbency",
                "Author’s prior reception",
                "Toxicity",
                "Positive sentiment",
                "Negative sentiment",
                "Lexical diversity",
                "Reading difficulty",
                "AQuA deliberative quality",
            ],
        )

    def test_reply_display_markers(self):
        self.assertEqual(
            REPLY_DISPLAY_MARKERS,
            {"loose": "o", "trees": "^", "hidden": "X"},
        )


if __name__ == "__main__":
    unittest.main()
