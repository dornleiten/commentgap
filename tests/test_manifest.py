from pathlib import Path
import tempfile
import unittest

from commentgap_scraper.manifest import Manifest
from commentgap_scraper.parsing import DiscoveredStory


class ManifestTests(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
