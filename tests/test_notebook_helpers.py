from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from commentgap_scraper.notebook_helpers import (
    parquet_files,
    parquet_relation,
    require_collection_credentials,
    run_scraper,
    sql_string_list,
)


class NotebookHelpersTests(unittest.TestCase):
    def test_parquet_helpers_discover_and_escape_paths(self):
        with tempfile.TemporaryDirectory() as directory:
            output_dir = Path(directory) / "data with 'quote'"
            expected = output_dir / "articles/year=2025/month=01/story.parquet"
            expected.parent.mkdir(parents=True)
            expected.touch()

            self.assertEqual(parquet_files(output_dir, "articles"), [expected])
            self.assertIsNone(parquet_relation(output_dir, "comments"))
            self.assertEqual(
                parquet_relation(output_dir, "articles"),
                "read_parquet(['"
                + str(expected).replace("'", "''")
                + "'], hive_partitioning=true, union_by_name=true)",
            )
            self.assertEqual(sql_string_list([expected]), "['" + str(expected).replace("'", "''") + "']")

    def test_credentials_require_contact_and_optional_hash_key(self):
        with self.assertRaisesRegex(RuntimeError, "COMMENTGAP_CONTACT"):
            require_collection_credentials("", environ={})
        with self.assertRaisesRegex(RuntimeError, "COMMENTGAP_HASH_KEY"):
            require_collection_credentials("contact@example.org", include_hash_key=True, environ={})
        require_collection_credentials(
            "contact@example.org",
            include_hash_key=True,
            environ={"COMMENTGAP_HASH_KEY": "test-key"},
        )

    @patch("commentgap_scraper.notebook_helpers.subprocess.run")
    def test_run_scraper_builds_the_same_command(self, run):
        run.return_value.returncode = 0
        result = run_scraper(
            "validate",
            project_root=Path("/project"),
            year=2025,
            output_dir=Path("/project/data"),
            python_executable="python-test",
        )
        self.assertEqual(result, 0)
        command = [
            "python-test",
            "-m",
            "commentgap_scraper",
            "validate",
            "--year",
            "2025",
            "--output",
            "/project/data",
        ]
        run.assert_called_once()
        self.assertEqual(run.call_args.args[0], command)
        self.assertEqual(run.call_args.kwargs["cwd"], Path("/project"))


if __name__ == "__main__":
    unittest.main()
