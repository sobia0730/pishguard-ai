"""Read and inspect the local LegitPhish v2 CSV."""

import argparse
from pathlib import Path

import pandas as pd


SOURCE_COLUMNS = {"URL": "url", "ClassLabel": "label"}
DEFAULT_CSV = Path("data/raw/url_features_extracted1.csv")


def load_legitphish(path: str | Path) -> pd.DataFrame:
    """Return url and label as nullable strings with source labels preserved."""
    with Path(path).open(encoding="utf-8-sig", newline="") as source:
        header = pd.read_csv(source, nrows=0)
        missing = SOURCE_COLUMNS.keys() - set(header.columns)
        if missing:
            raise ValueError(f"Missing required CSV columns: {', '.join(sorted(missing))}")

        source.seek(0)
        data = pd.read_csv(
            source,
            usecols=list(SOURCE_COLUMNS),
            dtype="string",
            keep_default_na=False,
            na_values=[""],
        )

    return data.rename(columns=SOURCE_COLUMNS)[list(SOURCE_COLUMNS.values())]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", nargs="?", type=Path, default=DEFAULT_CSV)
    args = parser.parse_args()

    try:
        data = load_legitphish(args.csv)
    except (OSError, ValueError) as error:
        parser.exit(1, f"Ingestion failed: {error}\n")

    counts = data["label"].value_counts()
    print(f"Rows: {len(data):,}")
    print(f"Columns: {', '.join(data.columns)}")
    print(f"Phishing (0): {counts.get('0', 0):,}")
    print(f"Legitimate (1): {counts.get('1', 0):,}")
    missing = data["label"].isna().sum()
    other = len(data) - missing - counts.get("0", 0) - counts.get("1", 0)
    print(f"Missing labels: {missing:,}")
    print(f"Other label values (rows): {other:,}")


if __name__ == "__main__":
    main()
