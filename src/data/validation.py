"""Validate raw URL records without making network requests."""

import argparse
import ipaddress
import json
import re
from pathlib import Path
from urllib.parse import urlsplit

import pandas as pd

from src.data.ingestion import DEFAULT_CSV, load_legitphish


CLEAN_CSV = Path("data/processed/urls_clean.csv")
QUALITY_REPORT = Path("reports/data_quality.json")
LABELS = {"0": "phishing", "1": "legitimate"}
INVALID_CHARACTERS = re.compile(r"[\s\x00-\x1f\x7f\\]")
DNS_LABEL = re.compile(r"[a-zA-Z0-9](?:[a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?")


def is_valid_url(url: str) -> bool:
    """Check HTTP(S) URL syntax using only the supplied text."""
    if INVALID_CHARACTERS.search(url):
        return False

    try:
        parsed = urlsplit(url)
        host = parsed.hostname
        port = parsed.port
    except ValueError:
        return False

    if parsed.scheme not in {"http", "https"} or not host or port == 0:
        return False

    try:
        ipaddress.ip_address(host)
    except ValueError:
        # A failed IP parse should not turn a malformed numeric address into a DNS name.
        try:
            domain = host.removesuffix(".").encode("idna").decode("ascii")
        except UnicodeError:
            return False
        labels = domain.split(".")
        return (
            len(domain) <= 253
            and len(labels) > 1
            and not labels[-1].isdigit()
            and all(DNS_LABEL.fullmatch(label) for label in labels)
        )

    return True


def validate_dataset(data: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Return clean URL/label rows and aggregate counts. Leave the input intact."""
    columns = ["url", "label"]
    if not data.columns.is_unique:
        raise ValueError("Dataset column names must be unique.")
    missing = set(columns) - set(data.columns)
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(sorted(missing))}")
    for column in columns:
        if not isinstance(data[column].dtype, pd.StringDtype):
            raise ValueError(f"Expected pandas string dtype for '{column}'.")

    rows = data[columns].apply(lambda column: column.str.strip())
    missing_url = rows["url"].isna() | rows["url"].eq("").fillna(False)
    missing_label = rows["label"].isna() | rows["label"].eq("").fillna(False)
    invalid_url = ~missing_url & ~rows["url"].fillna("").map(is_valid_url)
    invalid_label = ~missing_label & ~rows["label"].isin(LABELS)
    rejected = missing_url | missing_label | invalid_url | invalid_label
    eligible = rows.loc[~rejected]

    distinct_labels = eligible.groupby("url")["label"].nunique()
    conflicting_urls = distinct_labels[distinct_labels > 1].index
    conflict = eligible["url"].isin(conflicting_urls)
    consistent = eligible.loc[~conflict]
    clean = consistent.drop_duplicates(subset=columns).reset_index(drop=True)

    report = {
        "input_rows": len(data),
        "input_dtypes": {column: str(data[column].dtype) for column in columns},
        "missing_url_rows": int(missing_url.sum()),
        "missing_label_rows": int(missing_label.sum()),
        "invalid_url_rows": int(invalid_url.sum()),
        "invalid_label_rows": int(invalid_label.sum()),
        "invalid_or_missing_rows_removed": int(rejected.sum()),
        "conflicting_urls": len(conflicting_urls),
        "conflicting_rows_removed": int(conflict.sum()),
        "duplicate_rows_removed": len(consistent) - len(clean),
        "clean_rows": len(clean),
        "clean_label_counts": {
            name: int(clean["label"].eq(value).sum()) for value, name in LABELS.items()
        },
    }
    clean["label"] = clean["label"].astype("int8")
    return clean, report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", nargs="?", type=Path, default=DEFAULT_CSV)
    args = parser.parse_args()

    try:
        if args.csv.resolve() in {CLEAN_CSV.resolve(), QUALITY_REPORT.resolve()}:
            raise ValueError("Use a raw input path that differs from the output paths.")
        clean, report = validate_dataset(load_legitphish(args.csv))
        CLEAN_CSV.parent.mkdir(parents=True, exist_ok=True)
        QUALITY_REPORT.parent.mkdir(parents=True, exist_ok=True)
        clean.to_csv(CLEAN_CSV, index=False)
        QUALITY_REPORT.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    except (OSError, ValueError) as error:
        parser.exit(1, f"Validation failed: {error}\n")

    print(f"Retained {report['clean_rows']:,} of {report['input_rows']:,} rows")
    print(f"Clean CSV: {CLEAN_CSV}")
    print(f"Quality report: {QUALITY_REPORT}")


if __name__ == "__main__":
    main()
