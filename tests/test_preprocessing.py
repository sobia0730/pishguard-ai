import unittest
from unittest.mock import patch

import pandas as pd
import tldextract

from src.features.preprocessing import (
    DOMAIN_EXTRACTOR,
    FEATURE_COLUMNS,
    _url_features_and_group,
    extract_url_features,
    preprocess_dataset,
)


class PreprocessingTests(unittest.TestCase):
    def test_extracts_ordered_lexical_features(self):
        url = "https://shop.example.co.uk/login-login2@verify"
        result = extract_url_features(url)
        self.assertEqual(list(result), FEATURE_COLUMNS)
        self.assertEqual(result, {
            "url_length": len(url),
            "domain_length": 18,
            "dot_count": 3,
            "hyphen_count": 1,
            "digit_count": 1,
            "subdomain_count": 1,
            "has_ip_address": 0,
            "is_https": 1,
            "has_at_symbol": 1,
            "suspicious_keyword_count": 2,
        })

    def test_ip_and_malformed_urls(self):
        features = extract_url_features("http://[2001:db8::1]:8080/")
        self.assertEqual(features["has_ip_address"], 1)
        self.assertEqual(features["subdomain_count"], 0)
        self.assertEqual(features["is_https"], 0)
        with self.assertRaisesRegex(ValueError, "HTTP"):
            extract_url_features("example.test")

    def test_splits_are_reproducible_and_keep_related_hosts_together(self):
        # Each group has a distinct URL length to identify it in feature-only output.
        rows = [
            (f"https://{subdomain}.site{group:02}.co.uk/" + "x" * group, label)
            for group in range(20)
            for subdomain, label in (("a", 0), ("b", 1))
        ]
        data = pd.DataFrame(rows, columns=["url", "label"])
        splits, summary = preprocess_dataset(data)
        second, second_summary = preprocess_dataset(data)
        self.assertEqual(summary, second_summary)
        self.assertEqual(sum(len(frame) for frame in splits.values()), len(data))
        assigned = set()
        for name, frame in splits.items():
            pd.testing.assert_frame_equal(frame, second[name])
            self.assertEqual(list(frame), [*FEATURE_COLUMNS, "label"])
            self.assertTrue(frame["url_length"].value_counts().eq(2).all())
            lengths = set(frame["url_length"])
            self.assertFalse(assigned & lengths)
            assigned.update(lengths)

    def test_rejects_invalid_target(self):
        data = pd.DataFrame({"url": ["https://example.test"], "label": [2]})
        with self.assertRaisesRegex(ValueError, "Expected labels"):
            preprocess_dataset(data)

    def test_registered_domain_uses_registry_suffixes(self):
        examples = {
            "https://shop.example.co.uk": ("example.co.uk", 1),
            "http://auth.EXAMPLE.co.uk.:8080": ("example.co.uk", 1),
            "https://shop.another.co.uk": ("another.co.uk", 1),
            "https://one.blogspot.com": ("blogspot.com", 1),
            "https://two.blogspot.com": ("blogspot.com", 1),
            "https://one.site.test": ("site.test", 1),
            "http://[2001:0db8:0:0::1]": ("2001:db8::1", 0),
        }
        for url, (expected_group, expected_subdomains) in examples.items():
            with self.subTest(url=url):
                features, group = _url_features_and_group(url)
                self.assertEqual(group, expected_group)
                self.assertEqual(features["subdomain_count"], expected_subdomains)

    def test_cold_snapshot_load_needs_no_network(self):
        cold_extractor = tldextract.TLDExtract(
            suffix_list_urls=DOMAIN_EXTRACTOR.suffix_list_urls,
            cache_dir=None,
            fallback_to_snapshot=DOMAIN_EXTRACTOR.fallback_to_snapshot,
        )
        with patch("requests.sessions.Session.request", side_effect=AssertionError("Network request")), patch(
            "src.features.preprocessing.DOMAIN_EXTRACTOR", cold_extractor
        ):
            self.assertEqual(_url_features_and_group("https://a.example.co.uk")[1], "example.co.uk")

