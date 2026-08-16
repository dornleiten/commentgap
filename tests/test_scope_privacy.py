import json
from pathlib import Path
import tempfile
import unittest

import pyarrow as pa
import pyarrow.parquet as pq

from commentgap_scraper.cli import build_parser
from commentgap_scraper.config import ScrapeConfig
from commentgap_scraper.crawler import _verify_legacy_hash_key, discover
from commentgap_scraper.manifest import Manifest
from commentgap_scraper.privacy import AuthorPseudonymizer, ensure_hash_key_compatible


SITEMAP = """<?xml version="1.0" encoding="UTF-8"?>
<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
  <url>
    <loc>https://www.derstandard.at/story/3000000240001/december-example</loc>
    <lastmod>2024-12-15T12:00:00Z</lastmod>
  </url>
</urlset>
"""


class SitemapHttp:
    def __init__(self):
        self.urls = []

    def get_text(self, url):
        self.urls.append(url)
        return SITEMAP


class KeyVerificationApi:
    def __init__(self, postings):
        self.postings = postings

    def get_forum_info(self, _uri):
        return {"id": "forum", "stickyPostings": []}

    def get_threads_page(self, _forum_id):
        return {
            "edges": [{"node": posting} for posting in self.postings],
            "pageInfo": {"hasNextPage": False},
        }


class ScopePrivacyTests(unittest.TestCase):
    def test_cli_accepts_month_scope(self):
        parser = build_parser()
        args = parser.parse_args(
            ["crawl", "--year", "2024", "--month", "12", "--output", "dataset"]
        )
        self.assertEqual(args.month, 12)

    def test_discover_reads_and_exports_only_selected_month(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = ScrapeConfig(
                year=2024,
                month=12,
                output_dir=root,
                user_agent="CommentGap test",
                contact="test@example.org",
            )
            http = SitemapHttp()
            result = discover(config, http=http)
            self.assertEqual(http.urls, [
                "https://www.derstandard.at/sitemaps/sitemap-2024-12.xml"
            ])
            self.assertEqual(result, {"unique_stories": 1, "12": 1})
            with Manifest(root / "crawl_manifest.sqlite3") as manifest:
                self.assertEqual(len(manifest.rows(2024, 12)), 1)
                self.assertEqual(manifest.rows(2025), [])
            self.assertTrue(
                (root / "crawl_manifest/year=2024/month=12/manifest.parquet").exists()
            )

    def test_key_fingerprint_rejects_a_changed_key(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = ensure_hash_key_compatible(root, "original-key-0123456789")
            metadata = json.loads(path.read_text(encoding="utf-8"))
            self.assertNotIn("original-key", path.read_text(encoding="utf-8"))
            self.assertEqual(len(metadata["hash_key_fingerprint"]), 64)
            ensure_hash_key_compatible(root, "original-key-0123456789")
            with self.assertRaisesRegex(ValueError, "does not match"):
                ensure_hash_key_compatible(root, "different-key-012345678")

    def test_preexisting_dataset_requires_one_time_key_confirmation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            comment = root / "comments/year=2025/month=01/story.parquet"
            comment.parent.mkdir(parents=True)
            comment.touch()
            with self.assertRaisesRegex(ValueError, "--confirm-existing-hash-key"):
                ensure_hash_key_compatible(root, "original-key-0123456789")
            ensure_hash_key_compatible(
                root,
                "original-key-0123456789",
                confirm_existing=True,
            )
            self.assertTrue((root / "privacy_metadata.json").exists())

    def test_legacy_key_is_verified_against_existing_comment_hashes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            key = "original-key-0123456789"
            postings = [
                {
                    "id": f"comment-{index}",
                    "author": {"id": f"author-{index}"},
                    "legacy": {},
                    "replies": [],
                }
                for index in range(3)
            ]
            pseudonymizer = AuthorPseudonymizer(key)
            rows = [
                {
                    "story_id": "3000000250001",
                    "comment_id": posting["id"],
                    "author_hash": pseudonymizer.pseudonymize(posting),
                }
                for posting in postings
            ]
            path = root / "comments/year=2025/month=12/3000000250001.parquet"
            path.parent.mkdir(parents=True)
            pq.write_table(pa.Table.from_pylist(rows), path)
            api = KeyVerificationApi(postings)
            _verify_legacy_hash_key(
                root,
                api=api,
                pseudonymizer=AuthorPseudonymizer(key),
            )
            with self.assertRaisesRegex(ValueError, "does not reproduce"):
                _verify_legacy_hash_key(
                    root,
                    api=api,
                    pseudonymizer=AuthorPseudonymizer("different-key-012345678"),
                )


if __name__ == "__main__":
    unittest.main()
