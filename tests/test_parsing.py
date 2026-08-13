from pathlib import Path
import unittest

from commentgap_scraper.parsing import extract_page_config, parse_article, parse_sitemap


FIXTURES = Path(__file__).parent / "fixtures"


class ParsingTests(unittest.TestCase):
    def test_sitemap_filters_and_deduplicates_story_urls(self):
        stories = parse_sitemap((FIXTURES / "sitemap.xml").read_text(), 2025, 1)
        self.assertEqual(len(stories), 1)
        self.assertEqual(stories[0].story_id, "3000000255487")
        self.assertTrue(stories[0].url.endswith("example-one-updated"))

    def test_article_prefers_page_config_and_collects_body_and_sections(self):
        html = (FIXTURES / "article.html").read_text()
        result = parse_article(html, "https://fallback.invalid", "3000000256906")
        self.assertEqual(result["title"], "Configured title")
        self.assertEqual(result["subtitle"], "Configured subtitle")
        self.assertEqual(result["published_at"], "2025-02-11T10:00:00Z")
        self.assertEqual(result["section_1"], "Diskurs")
        self.assertEqual(result["section_2"], "Community")
        self.assertEqual(result["body"], "First paragraph.\n\nSecond paragraph with äöü.")

    def test_page_config_absent_is_safe(self):
        self.assertEqual(extract_page_config("<html></html>"), {})

    def test_consent_page_falls_back_to_sitemap_time_and_taxonomy(self):
        html = """
        <html><head><meta name="cXenseParse:taxonomy" content="inland/politik"></head>
        <body><article><h1 class="article-title">Title</h1>
        <div class="article-body"><p>Body</p></div></article></body></html>
        """
        result = parse_article(
            html,
            "https://www.derstandard.at/story/3000000000001/example",
            "3000000000001",
            "2025-03-11T15:00:00.000Z",
        )
        self.assertEqual(result["published_at"], "2025-03-11T15:00:00.000Z")
        self.assertEqual(result["published_at_source"], "sitemap_lastmod_fallback")
        self.assertEqual(result["section_1"], "/inland")
        self.assertEqual(result["section_2"], "/inland/politik")


if __name__ == "__main__":
    unittest.main()
