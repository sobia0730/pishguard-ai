"""Extract lexical URL features and create reproducible domain-grouped splits."""

import argparse
import ipaddress
import json
from pathlib import Path
from urllib.parse import urlsplit

import pandas as pd
import tldextract
from sklearn.model_selection import GroupShuffleSplit

from src.data.validation import CLEAN_CSV, is_valid_url


OUTPUT_DIR = Path("data/processed")
FEATURE_REPORT = Path("reports/feature_definitions.json")
SEED = 42
KEYWORDS = ("account", "confirm", "login", "password", "secure", "signin", "update", "verify")
DOMAIN_EXTRACTOR = tldextract.TLDExtract(
    suffix_list_urls=(),
    cache_dir=None,
    fallback_to_snapshot=True,
    include_psl_private_domains=False,
)
FEATURE_DEFINITIONS = {
    "url_length": "Characters in the trimmed URL.",
    "domain_length": "Characters in the lowercase IDNA hostname or normalized IP, excluding port.",
    "dot_count": "Literal dots in the URL.",
    "hyphen_count": "Literal hyphens in the URL.",
    "digit_count": "ASCII digits in the URL.",
    "subdomain_count": "Hostname labels before the registered domain. Unknown suffixes use the last two labels; IPs return zero.",
    "has_ip_address": "1 when the hostname is an IPv4 or IPv6 address, else 0.",
    "is_https": "1 for the HTTPS scheme, else 0.",
    "has_at_symbol": "1 when the URL contains a literal @, else 0.",
    "suspicious_keyword_count": "Number of distinct configured substrings in the lowercased URL.",
}
FEATURE_COLUMNS = list(FEATURE_DEFINITIONS)


def _url_features_and_group(url: str) -> tuple[dict[str, int], str]:
    url = url.strip()
    if not is_valid_url(url):
        raise ValueError("Expected a valid HTTP(S) URL.")
    parsed = urlsplit(url)
    host = parsed.hostname
    assert host is not None

    try:
        hostname = str(ipaddress.ip_address(host))
    except ValueError:
        hostname = host.removesuffix(".").encode("idna").decode("ascii").lower()
        labels = hostname.split(".")
        parts = DOMAIN_EXTRACTOR(hostname)
        group = parts.top_domain_under_public_suffix
        if group:
            subdomains = len(parts.subdomain.split(".")) if parts.subdomain else 0
        elif parts.suffix:
            group = hostname
            subdomains = 0
        else:
            # The PSL's implicit one-label rule for an unlisted suffix.
            group = ".".join(labels[-2:])
            subdomains = max(0, len(labels) - 2)
        has_ip = 0
    else:
        group = hostname
        subdomains = 0
        has_ip = 1

    lower_url = url.lower()
    features = {
        "url_length": len(url),
        "domain_length": len(hostname),
        "dot_count": url.count("."),
        "hyphen_count": url.count("-"),
        "digit_count": sum("0" <= character <= "9" for character in url),
        "subdomain_count": subdomains,
        "has_ip_address": has_ip,
        "is_https": int(parsed.scheme == "https"),
        "has_at_symbol": int("@" in url),
        "suspicious_keyword_count": sum(keyword in lower_url for keyword in KEYWORDS),
    }
    return features, group


def extract_url_features(url: str) -> dict[str, int]:
    """Extract the ordered model inputs from a single URL for inference."""
    features, _ = _url_features_and_group(url)
    return features


def preprocess_dataset(data: pd.DataFrame) -> tuple[dict[str, pd.DataFrame], dict]:
    """Convert Phase 2 data into feature-only splits plus the integer target."""
    splits, summary, _ = preprocess_with_positions(data)
    return splits, summary


def preprocess_with_positions(data: pd.DataFrame) -> tuple[dict[str, pd.DataFrame], dict, dict]:
    """Also return input row positions so Feast can recover URL record IDs."""
    if not {"url", "label"}.issubset(data.columns):
        raise ValueError("Expected url and label columns in the clean dataset.")
    if not data["label"].isin((0, 1)).all():
        raise ValueError("Expected labels 0 (phishing) and 1 (legitimate).")
    if not data["url"].map(lambda url: isinstance(url, str)).all():
        raise ValueError("Expected a URL string in each clean row.")

    extracted = [_url_features_and_group(url) for url in data["url"]]
    groups = pd.Series([group for _, group in extracted], dtype="string")
    if groups.nunique() < 4:
        raise ValueError("At least four domain groups are needed for the configured split.")
    features = pd.DataFrame(
        [values for values, _ in extracted], columns=FEATURE_COLUMNS, dtype="int32"
    )
    features["label"] = data["label"].to_numpy(dtype="int8")

    train, holdout = next(
        GroupShuffleSplit(test_size=0.30, random_state=SEED).split(features, groups=groups)
    )
    validation, test = next(
        GroupShuffleSplit(test_size=0.50, random_state=SEED + 1).split(
            features.iloc[holdout], groups=groups.iloc[holdout]
        )
    )
    indices = {"train": train, "validation": holdout[validation], "test": holdout[test]}
    splits = {}
    summary = {}
    assigned_groups = set()
    for name, positions in indices.items():
        frame = features.iloc[positions].reset_index(drop=True)
        split_groups = set(groups.iloc[positions])
        if assigned_groups & split_groups:
            raise ValueError("A domain group was assigned to more than one split.")
        assigned_groups.update(split_groups)
        counts = frame["label"].value_counts()
        if len(counts) != 2:
            raise ValueError(f"The {name} split needs examples from both classes.")
        splits[name] = frame
        summary[name] = {
            "rows": len(frame),
            "domain_groups": len(split_groups),
            "phishing": int(counts[0]),
            "legitimate": int(counts[1]),
        }
    return splits, summary, indices


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", nargs="?", type=Path, default=CLEAN_CSV)
    args = parser.parse_args()
    paths = {name: OUTPUT_DIR / f"{name}.csv" for name in ("train", "validation", "test")}

    try:
        if args.csv.resolve() in {path.resolve() for path in [*paths.values(), FEATURE_REPORT]}:
            raise ValueError("Use a clean input path that differs from the output paths.")
        data = pd.read_csv(args.csv, dtype={"url": "string", "label": "Int8"}, keep_default_na=False)
        splits, summary = preprocess_dataset(data)
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        FEATURE_REPORT.parent.mkdir(parents=True, exist_ok=True)
        for name, frame in splits.items():
            frame.to_csv(paths[name], index=False)
        report = {
            "features": [
                {"name": name, "dtype": "int32", "definition": definition}
                for name, definition in FEATURE_DEFINITIONS.items()
            ],
            "target": {"name": "label", "dtype": "int8", "0": "phishing", "1": "legitimate"},
            "suspicious_keywords": list(KEYWORDS),
            "grouping": {
                "parser": "tldextract",
                "parser_version": tldextract.__version__,
                "suffix_snapshot": "Bundled Public Suffix List (ICANN section)",
                "network_fetch": False,
                "cache": False,
                "registered_domain": "One label above the registry public suffix",
                "unknown_suffix": "Final two normalized DNS labels",
                "bare_public_suffix": "Normalized hostname",
                "ip_address": "Normalized IP address",
            },
            "input_rows": len(data),
            "split_seed": SEED,
            "holdout_split_seed": SEED + 1,
            "cross_split_domain_group_overlap": 0,
            "splits": summary,
        }
        FEATURE_REPORT.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    except (OSError, ValueError, TypeError) as error:
        parser.exit(1, f"Preprocessing failed: {error}\n")

    for name, stats in summary.items():
        print(f"{name}: {stats['rows']:,} rows, {stats['domain_groups']:,} groups, "
              f"{stats['phishing']:,} phishing, {stats['legitimate']:,} legitimate")
    print(f"Feature definitions and split report: {FEATURE_REPORT}")


if __name__ == "__main__":
    main()
