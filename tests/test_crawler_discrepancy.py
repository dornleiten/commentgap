import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import pyarrow.parquet as pq

from commentgap_scraper.config import ScrapeConfig
from commentgap_scraper.crawler import crawl, select_stratified_pilot
from commentgap_scraper.manifest import Manifest
from commentgap_scraper.parsing import DiscoveredStory
from commentgap_scraper.validation import validate_dataset


FIXTURES = Path(__file__).parent / "fixtures"


class StableMismatchHttp:
    def __init__(self):
        self.article_html = (FIXTURES / "article.html").read_text(encoding="utf-8")
        self.thread_requests = 0

    def get_text(self, _url):
        return self.article_html

    def post_json(self, _url, payload):
        if payload["operationName"] == "GetForumInfo":
            return {
                "data": {
                    "getForumByContextUri": {
                        "id": "forum-1",
                        "flags": [],
                        "metadata": [],
                        "totalPostingCount": 2,
                        "stickyPostings": [],
                    }
                }
            }
        self.thread_requests += 1
        return {
            "data": {
                "getForumRootPostingsV2": {
                    "pageInfo": {"hasNextPage": False, "nextCursor": None},
                    "edges": [
                        {
                            "cursor": "cursor-1",
                            "node": {
                                "id": "comment-1",
                                "lifecycleStatus": "Published",
                                "flags": [],
                                "rootPostingId": "comment-1",
                                "title": "Only returned comment",
                                "text": "Body",
                                "author": {"id": "author-1", "followerCount": 3},
                                "reactions": {"aggregated": []},
                                "history": {"created": "2025-01-01T13:00:00Z"},
                                "legacy": {"postingId": 1},
                                "replies": [],
                            },
                        }
                    ],
                }
            }
        }


class PilotApi:
    def __init__(self, info_by_story):
        self.info_by_story = info_by_story

    def get_forum_info(self, uri, **_kwargs):
        return self.info_by_story[uri.rsplit("/", 1)[-1]]


class CrawlerDiscrepancyTests(unittest.TestCase):
    def test_stratified_preflight_time_is_excluded_from_crawl_rate(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            story = DiscoveredStory(
                "3000000000001",
                2025,
                1,
                "https://www.derstandard.at/story/3000000000001/example",
                None,
            )
            with Manifest(root / "crawl_manifest.sqlite3") as manifest:
                manifest.upsert_discovered([story])
            config = ScrapeConfig(
                year=2025,
                output_dir=root,
                user_agent="CommentGap test",
                contact="test@example.org",
            )
            with (
                patch(
                    "commentgap_scraper.crawler.select_stratified_pilot",
                    side_effect=lambda candidates, **_kwargs: candidates,
                ),
                patch(
                    "commentgap_scraper.crawler.crawl_story",
                    return_value="completed",
                ),
                patch(
                    "commentgap_scraper.crawler.time.monotonic",
                    side_effect=[100.0, 600.0, 600.0, 660.0],
                ),
                self.assertLogs("commentgap_scraper", level="INFO") as logs,
            ):
                crawl(
                    config,
                    hash_key="0123456789abcdef-test-key",
                    limit=1,
                    stratified_pilot=True,
                    pilot_candidate_pool=1,
                    http=StableMismatchHttp(),
                )

            output = "\n".join(logs.output)
            self.assertIn("pilot preflight complete: 1 stories selected | elapsed 8m 20s", output)
            self.assertIn("elapsed 1m 00s", output)
            self.assertIn("60.0 stories/hour", output)

    def test_stratified_pilot_balances_months_and_records_selection(self):
        with tempfile.TemporaryDirectory() as directory:
            candidates = []
            info = {}
            counts = [None, 0, 10, 100, 500, 1_500]
            for rank, count in enumerate(counts):
                for month in range(1, 13):
                    story_id = f"3{rank:02d}{month:02d}00000000"
                    candidates.append(
                        {
                            "story_id": story_id,
                            "year": 2025,
                            "month": month,
                            "url": f"https://example/{story_id}",
                        }
                    )
                    info[story_id] = (
                        None
                        if count is None
                        else {
                            "id": f"forum-{story_id}",
                            "totalPostingCount": count,
                        }
                    )
            selected = select_stratified_pilot(
                candidates,
                api=PilotApi(info),
                limit=12,
                seed=7,
                output_dir=Path(directory),
            )
            self.assertEqual(len(selected), 12)
            self.assertEqual({row["month"] for row in selected}, set(range(1, 13)))
            audit = (Path(directory) / "pilot_selection.json").read_text(encoding="utf-8")
            self.assertIn('"selected_count": 12', audit)
            self.assertIn('"1000+"', audit)

    def test_count_mismatch_is_retained_without_recrawling_threads(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            story = DiscoveredStory(
                "3000000000001",
                2025,
                1,
                "https://www.derstandard.at/story/3000000000001/example",
                "2025-01-01T12:00:00Z",
            )
            with Manifest(root / "crawl_manifest.sqlite3") as manifest:
                manifest.upsert_discovered([story])
            config = ScrapeConfig(
                year=2025,
                output_dir=root,
                user_agent="CommentGap test",
                contact="test@example.org",
            )
            http = StableMismatchHttp()
            result = crawl(
                config,
                hash_key="0123456789abcdef-test-key",
                http=http,
            )
            self.assertEqual(result, {"completed_with_count_discrepancy": 1})
            self.assertEqual(http.thread_requests, 1)

            with Manifest(root / "crawl_manifest.sqlite3") as manifest:
                row = manifest.story(story.story_id)
            self.assertEqual(row["status"], "completed_with_count_discrepancy")
            self.assertEqual(row["attempts"], 1)
            self.assertEqual(row["expected_count"], 2)
            self.assertEqual(row["observed_count"], 1)
            self.assertIsNone(row["error_category"])

            forum_path = root / "forums/year=2025/month=01/3000000000001.parquet"
            forum = pq.read_table(forum_path).to_pylist()[0]
            self.assertEqual(forum["observed_unique_count"], 1)
            self.assertEqual(forum["observed_published_count"], 1)
            self.assertEqual(forum["observed_deleted_count"], 0)
            self.assertEqual(forum["posting_count_difference"], -1)
            self.assertEqual(forum["posting_count_discrepancy_absolute"], 1)
            self.assertEqual(forum["posting_count_discrepancy_pct"], 50.0)
            self.assertIsNone(forum["count_discrepancy_reproduced"])
            self.assertEqual(forum["pagination_page_count"], 1)
            self.assertEqual(forum["root_edge_count"], 1)
            self.assertEqual(forum["flattened_record_count"], 1)
            self.assertTrue(forum["cursor_walk_complete"])
            self.assertTrue(forum["cursor_progression_valid"])

            comment_path = root / "comments/year=2025/month=01/3000000000001.parquet"
            comment = pq.read_table(comment_path).to_pylist()[0]
            self.assertEqual(comment["effective_text"], "Only returned comment\nBody")
            page_path = root / "forum_pages/year=2025/month=01/3000000000001.parquet"
            pages = pq.read_table(page_path).to_pylist()
            self.assertEqual([page["page_kind"] for page in pages], ["sticky", "threads"])
            self.assertEqual(pages[1]["root_edge_count"], 1)

            summary = validate_dataset(root, 2025)
            self.assertTrue(summary["passed"])
            self.assertEqual(summary["forum_count_mismatches"], 1)
            self.assertEqual(summary["unexpected_forum_count_mismatches"], 0)
            self.assertEqual(summary["reproduced_count_discrepancy_forums"], 0)
            self.assertEqual(summary["count_discrepancy_forums_not_recrawled"], 1)
            self.assertEqual(summary["comments_without_forum_rows"], 0)
            self.assertEqual(summary["sticky_descendant_rows"], 0)
            self.assertEqual(summary["forum_page_rows"], 1)
            self.assertEqual(summary["forum_page_aggregate_mismatches"], 0)

            metadata = json.loads(
                (
                    root / "collection_metadata/year=2025/metadata.json"
                ).read_text(encoding="utf-8")
            )
            self.assertEqual(metadata["schema_version"], 3)
            self.assertIn("last_crawl_finished_at", metadata)


if __name__ == "__main__":
    unittest.main()
