from pathlib import Path
import unittest

from commentgap_scraper.config import ScrapeConfig


class ConfigTests(unittest.TestCase):
    def test_month_scope_is_validated(self):
        with self.assertRaisesRegex(ValueError, "month must be between 1 and 12"):
            ScrapeConfig(
                year=2024,
                month=13,
                output_dir=Path("output"),
                user_agent="CommentGap test",
                contact="test@example.org",
            )

    def test_request_interval_cannot_undercut_published_crawl_delay(self):
        with self.assertRaisesRegex(ValueError, "at least one second"):
            ScrapeConfig(
                year=2025,
                output_dir=Path("output"),
                user_agent="CommentGap test",
                contact="test@example.org",
                request_interval=0.5,
            )


if __name__ == "__main__":
    unittest.main()
