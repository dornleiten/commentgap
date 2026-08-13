from pathlib import Path
import tempfile
import unittest

from commentgap_scraper.manifest import Manifest
from commentgap_scraper.parsing import DiscoveredStory


class ManifestTests(unittest.TestCase):
    def test_monthly_random_selection_is_seeded_and_month_balanced(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "manifest.sqlite3"
            stories = [
                DiscoveredStory(
                    f"30000000{month:02d}{index:03d}",
                    2025,
                    month,
                    f"https://example/story/{month}/{index}",
                    None,
                )
                for month in (1, 2, 3)
                for index in range(20)
            ]
            with Manifest(path) as manifest:
                manifest.upsert_discovered(stories)
                first = manifest.stories_for_crawl(
                    2025, limit=12, monthly_random=True, selection_seed=41
                )
                repeated = manifest.stories_for_crawl(
                    2025, limit=12, monthly_random=True, selection_seed=41
                )
                different = manifest.stories_for_crawl(
                    2025, limit=12, monthly_random=True, selection_seed=42
                )
            self.assertEqual(
                [row["story_id"] for row in first],
                [row["story_id"] for row in repeated],
            )
            self.assertNotEqual(
                [row["story_id"] for row in first],
                [row["story_id"] for row in different],
            )
            self.assertEqual(
                {month: sum(row["month"] == month for row in first) for month in (1, 2, 3)},
                {1: 4, 2: 4, 3: 4},
            )

    def test_count_discrepancy_is_a_terminal_nonfailure(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "manifest.sqlite3"
            story = DiscoveredStory(
                "3000000000002", 2025, 1, "https://example/story/2", None
            )
            with Manifest(path) as manifest:
                manifest.upsert_discovered([story])
                manifest.mark_terminal(
                    story.story_id,
                    "completed_with_count_discrepancy",
                    "finish",
                    forum_id="forum",
                    expected_count=10,
                    observed_count=9,
                )
                row = manifest.story(story.story_id)
                self.assertEqual(row["status"], "completed_with_count_discrepancy")
                self.assertEqual(row["pagination_complete"], 1)
                self.assertEqual(manifest.stories_for_crawl(2025), [])
                self.assertEqual(manifest.stories_for_crawl(2025, retry_failed=True), [])

    def test_checkpoint_survives_reopen_and_terminal_status_is_resumable(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "manifest.sqlite3"
            story = DiscoveredStory("3000000000001", 2025, 1, "https://example/story/1", None)
            with Manifest(path) as manifest:
                manifest.upsert_discovered([story])
                manifest.mark_in_progress(story.story_id, "start")
                manifest.save_progress(
                    story.story_id,
                    forum_id="forum",
                    expected_count=100,
                    next_cursor="cursor-2",
                    page_index=3,
                )
            with Manifest(path) as manifest:
                row = manifest.stories_for_crawl(2025)[0]
                self.assertEqual(row["next_cursor"], "cursor-2")
                self.assertEqual(row["page_index"], 3)
                self.assertEqual(row["forum_id"], "forum")
                manifest.mark_terminal(
                    story.story_id,
                    "failed",
                    "finish",
                    error_category="test",
                    error_message="safe error",
                )
                self.assertEqual(manifest.status_counts(2025), {"failed": 1})
                self.assertEqual(manifest.stories_for_crawl(2025), [])
                self.assertEqual(len(manifest.stories_for_crawl(2025, retry_failed=True)), 1)

    def test_only_failed_excludes_pending_and_includes_interrupted(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "manifest.sqlite3"
            stories = [
                DiscoveredStory(
                    f"300000000000{index}",
                    2025,
                    1,
                    f"https://example/story/{index}",
                    None,
                )
                for index in range(1, 4)
            ]
            with Manifest(path) as manifest:
                manifest.upsert_discovered(stories)
                manifest.mark_terminal(stories[1].story_id, "failed", "finish")
                manifest.mark_in_progress(stories[2].story_id, "start")
                selected = manifest.stories_for_crawl(2025, only_failed=True)
            self.assertEqual(
                {row["story_id"] for row in selected},
                {stories[1].story_id, stories[2].story_id},
            )


if __name__ == "__main__":
    unittest.main()
