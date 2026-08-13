import json
from pathlib import Path
import unittest

from commentgap_scraper.privacy import AuthorPseudonymizer
from commentgap_scraper.transform import flatten_postings, merge_comment_records
from commentgap_scraper.validation import validate_row_invariants


FIXTURES = Path(__file__).parent / "fixtures"


class TransformTests(unittest.TestCase):
    def setUp(self):
        self.roots = json.loads((FIXTURES / "postings.json").read_text())
        self.pseudonymizer = AuthorPseudonymizer("0123456789abcdef-test-key")

    def test_nested_tree_and_reactions_are_flattened(self):
        rows = flatten_postings(
            self.roots,
            story_id="3000000256906",
            forum_id="forum-1",
            pseudonymizer=self.pseudonymizer,
            collected_at="2026-08-13T12:00:00Z",
            page_index=1,
        )
        self.assertEqual([row["depth"] for row in rows], [0, 1, 2])
        self.assertEqual(rows[1]["parent_comment_id"], "root-1")
        self.assertEqual(rows[2]["root_comment_id"], "root-1")
        self.assertEqual(rows[0]["votes_positive"], 12)
        self.assertEqual(rows[0]["votes_negative"], 3)
        self.assertFalse(rows[0]["is_leaf"])
        self.assertTrue(rows[2]["is_leaf"])

    def test_raw_author_identifiers_are_not_persisted(self):
        rows = flatten_postings(
            self.roots,
            story_id="s",
            forum_id="f",
            pseudonymizer=self.pseudonymizer,
            collected_at="now",
            page_index=1,
        )
        serialized = json.dumps(rows)
        self.assertNotIn("Visible Name", serialized)
        self.assertNotIn("private-author-id", serialized)
        self.assertNotIn("raw-deleted-uuid", serialized)
        self.assertRegex(rows[0]["author_hash"], r"^author_[0-9a-f]{64}$")
        self.assertRegex(rows[1]["author_hash"], r"^author_[0-9a-f]{64}$")

    def test_sticky_duplicate_is_merged(self):
        regular = flatten_postings(
            self.roots,
            story_id="s",
            forum_id="f",
            pseudonymizer=self.pseudonymizer,
            collected_at="now",
            page_index=1,
        )
        sticky = flatten_postings(
            self.roots,
            story_id="s",
            forum_id="f",
            pseudonymizer=self.pseudonymizer,
            collected_at="now",
            page_index=0,
            is_sticky=True,
        )
        merged = merge_comment_records(sticky + regular)
        self.assertEqual(len(merged), 3)
        self.assertTrue(all(row["is_sticky"] for row in merged))
        self.assertEqual([row["display_order"] for row in merged], [1, 2, 3])
        self.assertEqual(validate_row_invariants(merged), [])

    def test_pseudonym_is_stable_and_keyed(self):
        first = self.pseudonymizer.pseudonymize(self.roots[0])
        second = self.pseudonymizer.pseudonymize(self.roots[0])
        other = AuthorPseudonymizer("different-secret-key-value").pseudonymize(self.roots[0])
        self.assertEqual(first, second)
        self.assertNotEqual(first, other)


if __name__ == "__main__":
    unittest.main()
