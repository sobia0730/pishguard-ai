import unittest

import pandas as pd

from src.data.validation import is_valid_url, validate_dataset


class ValidationTests(unittest.TestCase):
    def test_accepts_http_https_and_ip_addresses(self):
        for url in (
            "https://example.test/path?x=1#part",
            "HTTP://example.test:8080",
            "https://mañana.example/",
            "http://192.0.2.1",
            "https://[2001:db8::1]:443/",
        ):
            with self.subTest(url=url):
                self.assertTrue(is_valid_url(url))

    def test_rejects_malformed_and_non_http_urls(self):
        for url in (
            "example.test",
            "ftp://example.test",
            "https:///path",
            "https://example.test:65536",
            "https://example.test/a b",
            "https://exa\nmple.test",
            "https://example.test\\path",
            "https://-bad.example",
            "http://192.0.2.999",
            "https://[2001:db8::1",
        ):
            with self.subTest(url=url):
                self.assertFalse(is_valid_url(url))

    def test_validation_cleans_and_counts_rejections(self):
        data = pd.DataFrame(
            [
                (" https://keep.example ", " 0 "),
                ("https://keep.example", "0"),
                ("https://other.example", "1"),
                ("https://conflict.example", "0"),
                ("https://conflict.example", "1"),
                (None, " "),
                ("bad", "unknown"),
                ("https://missing-label.example", None),
            ],
            columns=["url", "label"],
            dtype="string",
        )
        before = data.copy()

        clean, report = validate_dataset(data)

        pd.testing.assert_frame_equal(data, before)
        self.assertEqual(clean["url"].tolist(), ["https://keep.example", "https://other.example"])
        self.assertEqual(clean["label"].tolist(), [0, 1])
        self.assertEqual(str(clean["label"].dtype), "int8")
        self.assertEqual(report["missing_url_rows"], 1)
        self.assertEqual(report["missing_label_rows"], 2)
        self.assertEqual(report["invalid_url_rows"], 1)
        self.assertEqual(report["invalid_label_rows"], 1)
        self.assertEqual(report["invalid_or_missing_rows_removed"], 3)
        self.assertEqual(report["conflicting_urls"], 1)
        self.assertEqual(report["conflicting_rows_removed"], 2)
        self.assertEqual(report["duplicate_rows_removed"], 1)
        self.assertEqual(report["clean_label_counts"], {"phishing": 1, "legitimate": 1})

    def test_requires_columns_and_string_dtypes(self):
        with self.assertRaisesRegex(ValueError, "Missing required columns: label"):
            validate_dataset(pd.DataFrame({"url": ["https://example.test"]}, dtype="string"))

        with self.assertRaisesRegex(ValueError, "string dtype for 'label'"):
            validate_dataset(pd.DataFrame({
                "url": pd.Series(["https://example.test"], dtype="string"),
                "label": [1],
            }))
