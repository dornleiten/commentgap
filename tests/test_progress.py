import unittest

from commentgap_scraper.progress import (
    crawl_progress_line,
    format_duration,
    forum_progress_line,
)


class ProgressTests(unittest.TestCase):
    def test_duration_and_crawl_eta_are_human_readable(self):
        self.assertEqual(format_duration(3_661), "1h 01m 01s")
        line = crawl_progress_line(25, 100, 3_600, {"completed": 24, "no_forum": 1})
        self.assertIn("25/100 (25.0%)", line)
        self.assertIn("ETA 3h 00m 00s", line)
        self.assertIn("25.0 stories/hour", line)
        self.assertIn("completed=24", line)

    def test_forum_progress_caps_percentage_when_tombstones_are_retained(self):
        line = forum_progress_line("story", 15, 67, 66, complete=True)
        self.assertIn("67/66 records (100.0%)", line)
        self.assertTrue(line.endswith("complete"))


if __name__ == "__main__":
    unittest.main()
