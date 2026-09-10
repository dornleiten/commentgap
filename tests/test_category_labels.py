import unittest

from commentgap_analysis.category_labels import translate_news_category


class NewsCategoryLabelTests(unittest.TestCase):
    def test_legacy_categories_accept_stored_path_values(self):
        self.assertEqual(translate_news_category("/inland"), "domestic")
        self.assertEqual(translate_news_category("wissenschaft"), "science")

    def test_unmapped_categories_are_left_unchanged(self):
        self.assertEqual(translate_news_category("/imfokus"), "/imfokus")
        self.assertEqual(translate_news_category("/besserleben"), "/besserleben")

    def test_missing_values_are_left_unchanged(self):
        self.assertIsNone(translate_news_category(None))


if __name__ == "__main__":
    unittest.main()
