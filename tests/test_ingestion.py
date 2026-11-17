import io
import sys
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from src.data.ingestion import load_legitphish, main


class IngestionTests(unittest.TestCase):
    def setUp(self):
        directory = TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.csv = Path(directory.name) / "sample.csv"

    def test_load_preserves_raw_values_and_ignores_supplied_features(self):
        self.csv.write_text(
            "ClassLabel,URL,url_length\n"
            "0,https://a.example.test,22\n"
            "1,https://b.example.test,22\n"
            "unknown,NA,2\n"
            ",,0\n",
            encoding="utf-8-sig",
        )

        data = load_legitphish(self.csv)

        self.assertEqual(list(data.columns), ["url", "label"])
        self.assertEqual(list(data["label"].iloc[:3]), ["0", "1", "unknown"])
        self.assertEqual(data["url"].iloc[2], "NA")
        self.assertEqual(data.isna().sum().to_dict(), {"url": 1, "label": 1})

    def test_load_requires_source_label_header(self):
        self.csv.write_text("URL,label\nhttps://a.example.test,0\n", encoding="utf-8")

        with self.assertRaisesRegex(ValueError, "ClassLabel"):
            load_legitphish(self.csv)

    def test_cli_prints_counts_without_url_values(self):
        self.csv.write_text(
            "URL,ClassLabel\nhttps://a.example.test,0\n"
            "https://b.example.test,1\nhttps://c.example.test,unknown\n"
            "https://d.example.test,\n",
            encoding="utf-8",
        )

        with patch.object(sys, "argv", ["ingestion", str(self.csv)]):
            with redirect_stdout(io.StringIO()) as output:
                main()

        self.assertEqual(
            output.getvalue().splitlines(),
            [
                "Rows: 4",
                "Columns: url, label",
                "Phishing (0): 1",
                "Legitimate (1): 1",
                "Missing labels: 1",
                "Other label values (rows): 1",
            ],
        )

    def test_cli_reports_missing_file(self):
        with patch.object(sys, "argv", ["ingestion", str(self.csv)]):
            with redirect_stderr(io.StringIO()) as output, self.assertRaises(SystemExit) as error:
                main()

        self.assertEqual(error.exception.code, 1)
        self.assertIn("Ingestion failed:", output.getvalue())
