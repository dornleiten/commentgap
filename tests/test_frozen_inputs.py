"""Report missing saved inputs without fabricating notebook results."""
import unittest
from unittest.mock import patch
from io import StringIO

from commentgap_analysis.frozen_inputs import unavailable


class FrozenInputTests(unittest.TestCase):
    def test_unavailable_identifies_variable_and_required_saved_input(self):
        with patch("sys.stdout", new_callable=StringIO) as output:
            unavailable("coordinates", "saved/umap_coordinates.parquet", "UMAP is not rerun.")
        message = output.getvalue()
        self.assertIn("coordinates", message)
        self.assertIn("saved/umap_coordinates.parquet", message)
        self.assertIn("UMAP is not rerun", message)
