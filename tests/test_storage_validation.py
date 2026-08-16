import json
from pathlib import Path
import tempfile
import unittest

import pyarrow.parquet as pq
import pyarrow as pa

from commentgap_scraper.legacy import export_legacy
from commentgap_scraper.crawler import _resume_is_consistent
from commentgap_scraper.manifest import Manifest
from commentgap_scraper.migration import migrate_existing
from commentgap_scraper.parsing import DiscoveredStory
from commentgap_scraper.storage import ParquetStore
from commentgap_scraper.validation import validate_dataset


class StorageValidationTests(unittest.TestCase):
    def test_offline_migration_adds_effective_text_and_clears_sticky_reply(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "comments/year=2025/month=01/story.parquet"
            path.parent.mkdir(parents=True)
            pq.write_table(
                pa.Table.from_pylist(
                    [
                        {
                            "comment_id": "reply",
                            "story_id": "story",
                            "forum_id": "forum",
                            "year": 2025,
                            "month": 1,
                            "depth": 1,
                            "title": "Heading",
                            "text": "Comment",
                            "is_sticky": True,
                        }
                    ]
                ),
                path,
            )
            result = migrate_existing(root, 2025)
            row = pq.read_table(path).to_pylist()[0]
            self.assertEqual(row["effective_text"], "Heading\nComment")
            self.assertFalse(row["is_sticky"])
            self.assertEqual(result["sticky_descendants_cleared"], 1)

    def test_offline_migration_preserves_evidenced_sticky_reply(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "comments/year=2025/month=01/story.parquet"
            path.parent.mkdir(parents=True)
            pq.write_table(
                pa.Table.from_pylist(
                    [
                        {
                            "comment_id": "reply",
                            "story_id": "story",
                            "forum_id": "forum",
                            "year": 2025,
                            "month": 1,
                            "depth": 1,
                            "is_sticky": True,
                        }
                    ]
                ),
                path,
            )
            diagnostic = root / "forum_pages/year=2025/month=01/story.parquet"
            diagnostic.parent.mkdir(parents=True)
            diagnostic.touch()
            result = migrate_existing(root, 2025)
            row = pq.read_table(path).to_pylist()[0]
            self.assertTrue(row["is_sticky"])
            self.assertEqual(result["sticky_descendants_cleared"], 0)

    def test_resume_prunes_only_an_uncommitted_page(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            store = ParquetStore(root)
            for page_index in range(3):
                store.write_comment_page("story", page_index, [])
            story = {"story_id": "story", "forum_id": "forum", "page_index": 2}
            self.assertTrue(_resume_is_consistent(store, story, "forum"))
            self.assertEqual(
                [path.name for path in store.staging_pages("story")],
                ["page-000000.parquet", "page-000001.parquet"],
            )

    def test_parquet_finalize_validation_and_legacy_export(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            store = ParquetStore(root)
            article = {
                "story_id": "3000000000001",
                "year": 2025,
                "month": 1,
                "canonical_url": "https://www.derstandard.at/story/3000000000001/example",
                "published_at": "2025-01-01T12:00:00Z",
                "published_at_source": "page",
                "modified_at": None,
                "sitemap_lastmod": "2025-01-01T12:00:00Z",
                "title": "Article",
                "subtitle": "Subtitle",
                "body": "Body",
                "section_1": "Inland",
                "section_2": None,
                "section_3": None,
                "collected_at": "2026-08-13T12:00:00Z",
            }
            store.write_article(article)
            base = {
                "legacy_posting_id": "123",
                "story_id": article["story_id"],
                "forum_id": "forum",
                "depth": 0,
                "root_order": 100000,
                "preorder_position": 1000000,
                "display_order": None,
                "is_root": True,
                "is_leaf": False,
                "created_at": "2025-01-01T13:00:00Z",
                "title": "Comment",
                "text": "Text",
                "lifecycle_status": "Published",
                "flags_json": "[]",
                "is_sticky": False,
                "votes_positive": 3,
                "votes_negative": 1,
                "author_hash": "author_" + "a" * 64,
                "author_follower_count": 2,
                "collected_at": "2026-08-13T12:00:00Z",
            }
            root_comment = {
                **base,
                "comment_id": "comment-root",
                "parent_comment_id": None,
                "root_comment_id": "comment-root",
            }
            child = {
                **base,
                "comment_id": "comment-child",
                "legacy_posting_id": "124",
                "parent_comment_id": "comment-root",
                "root_comment_id": "comment-root",
                "depth": 1,
                "preorder_position": 1000001,
                "is_root": False,
                "is_leaf": True,
            }
            store.write_comment_page(article["story_id"], 0, [{**root_comment, "is_sticky": True}])
            store.write_comment_page(article["story_id"], 1, [root_comment, child])
            _, count, reconciled, deleted = store.finalize_comments(
                article["story_id"], 2025, 1
            )
            self.assertEqual(count, 2)
            self.assertEqual(reconciled, 2)
            self.assertEqual(deleted, 0)
            store.write_forum(
                {
                    "story_id": article["story_id"],
                    "year": 2025,
                    "month": 1,
                    "forum_id": "forum",
                    "flags_json": "[]",
                    "metadata_json": "[]",
                    "reported_posting_count": 2,
                    "observed_unique_count": 2,
                    "observed_published_count": 2,
                    "observed_deleted_count": 0,
                    "reconciled_posting_count": 2,
                    "posting_count_difference": 0,
                    "posting_count_discrepancy_absolute": 0,
                    "posting_count_discrepancy_pct": 0.0,
                    "count_discrepancy_reproduced": None,
                    "crawl_status": "completed",
                    "collected_at": "2026-08-13T12:00:00Z",
                }
            )
            with Manifest(root / "crawl_manifest.sqlite3") as manifest:
                manifest.upsert_discovered(
                    [
                        DiscoveredStory(
                            article["story_id"], 2025, 1, article["canonical_url"], None
                        )
                    ]
                )
                manifest.mark_terminal(
                    article["story_id"],
                    "completed",
                    "2026-08-13T12:01:00Z",
                    forum_id="forum",
                    expected_count=2,
                    observed_count=2,
                )
                store.export_manifest(manifest.rows(2025), 2025)

            summary = validate_dataset(root, 2025)
            self.assertTrue(summary["passed"], json.dumps(summary, indent=2))
            self.assertEqual(summary["comment_rows"], 2)
            self.assertEqual(summary["missing_root_rows"], 0)
            self.assertEqual(summary["invalid_tree_relationship_rows"], 0)
            self.assertEqual(summary["monthly_comment_rows"], {"01": 2})
            self.assertEqual(summary["raw_author_identifier_columns"], [])
            self.assertEqual(summary["comments_without_forum_rows"], 0)
            self.assertEqual(summary["unexpected_forum_count_mismatches"], 0)
            monthly_summary = validate_dataset(root, 2025, month=1)
            self.assertTrue(monthly_summary["passed"], json.dumps(monthly_summary, indent=2))
            self.assertEqual(monthly_summary["month"], 1)
            self.assertTrue(
                (root / "qa_summary/year=2025/month=01/summary.json").exists()
            )
            comments_path, articles_path = export_legacy(root, 2025)
            self.assertTrue(comments_path.exists())
            self.assertTrue(articles_path.exists())
            legacy_rows = pq.read_table(comments_path).to_pylist()
            self.assertEqual(
                legacy_rows[0]["heading_and_text.comment"], "Comment\nText"
            )


if __name__ == "__main__":
    unittest.main()
