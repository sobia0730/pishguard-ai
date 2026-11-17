"""Build a local Feast snapshot and demonstrate retrieval without visiting URLs."""

import argparse
from datetime import timedelta
from pathlib import Path

import pandas as pd
from feast import FeatureStore

from feature_store.feature_repo.entities import URL_ID, url_record, url_record_id
from feature_store.feature_repo.feature_views import FEATURE_REFS, VIEW_NAME, make_url_feature_view
from feature_store.feature_repo.features import EVENT_TIMESTAMP, FEATURE_FILE
from src.data.validation import CLEAN_CSV
from src.features.preprocessing import FEATURE_COLUMNS, OUTPUT_DIR, extract_url_features, preprocess_with_positions


REPO_PATH = Path(__file__).resolve().parent / "feature_repo"
SNAPSHOT_TIME = pd.Timestamp("2025-05-22T00:00:00Z").as_unit("ms")
ENTITY_FILE = "split_entities.parquet"


def prepare_snapshot(clean: pd.DataFrame, split_dir: Path, repo_path: Path) -> pd.DataFrame:
    """Recover URL IDs in the original split order and verify the saved features."""
    splits, _, positions = preprocess_with_positions(clean)
    features = []
    entities = []
    for name, split in splits.items():
        saved = pd.read_csv(
            split_dir / f"{name}.csv",
            dtype={**dict.fromkeys(FEATURE_COLUMNS, "int32"), "label": "int8"},
        )
        if not split.equals(saved):
            raise ValueError(f"Phase 3 split mismatch: {name}. Check the clean CSV and saved split.")

        ids = clean["url"].iloc[positions[name]].map(url_record_id).reset_index(drop=True)
        identity = pd.DataFrame({URL_ID: ids, EVENT_TIMESTAMP: SNAPSHOT_TIME})
        features.append(identity.assign(**split[FEATURE_COLUMNS]))
        entities.append(identity.assign(split=name, label=split["label"]))

    feature_rows = pd.concat(features, ignore_index=True)
    entity_rows = pd.concat(entities, ignore_index=True)
    if not feature_rows[URL_ID].is_unique:
        raise ValueError("A URL record ID was repeated in the feature snapshot.")
    if feature_rows.isna().any().any() or entity_rows.isna().any().any():
        raise ValueError("Snapshot rows must have complete features and entity metadata.")

    data_dir = repo_path / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    feature_rows.to_parquet(data_dir / FEATURE_FILE, index=False)
    entity_rows.to_parquet(data_dir / ENTITY_FILE, index=False)
    return entity_rows


def register_features(repo_path: Path = REPO_PATH) -> FeatureStore:
    store = FeatureStore(repo_path=str(repo_path))
    store.apply([url_record, make_url_feature_view(repo_path / "data" / FEATURE_FILE)])
    return store


def retrieve_historical(store: FeatureStore, split_entities: pd.DataFrame, split: str = "train") -> pd.DataFrame:
    """Point-in-time join for one existing split, in Phase 3 row order."""
    request = split_entities.loc[split_entities["split"].eq(split), [URL_ID, EVENT_TIMESTAMP, "label"]]
    if request.empty:
        raise ValueError(f"No entity rows for split '{split}'.")
    result = store.get_historical_features(entity_df=request, features=FEATURE_REFS).to_df()
    result = result.set_index(URL_ID).reindex(request[URL_ID])
    if result[[*FEATURE_COLUMNS, "label"]].isna().any().any():
        raise ValueError(f"Feast returned missing features or labels for '{split}'.")
    return result[[*FEATURE_COLUMNS, "label"]].astype(
        {**dict.fromkeys(FEATURE_COLUMNS, "int32"), "label": "int8"}
    ).reset_index(drop=True)


def materialize_online(store: FeatureStore) -> None:
    store.materialize(
        start_date=(SNAPSHOT_TIME - timedelta(seconds=1)).to_pydatetime(),
        end_date=(SNAPSHOT_TIME + timedelta(seconds=1)).to_pydatetime(),
        feature_views=[VIEW_NAME],
    )


def retrieve_online(store: FeatureStore, url: str) -> dict[str, int]:
    """Return the stored feature vector for a known URL or raise on a miss."""
    result = store.get_online_features(
        features=FEATURE_REFS, entity_rows=[{URL_ID: url_record_id(url)}]
    ).to_dict()
    values = {name: result[name][0] for name in FEATURE_COLUMNS}
    if any(value is None for value in values.values()):
        raise LookupError("No complete online feature vector for this URL record.")
    return values


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-online", action="store_true", help="Run the historical demo only.")
    parser.add_argument("--prepare-only", action="store_true", help="Write only the offline snapshot files.")
    args = parser.parse_args()

    clean = pd.read_csv(CLEAN_CSV, dtype={"url": "string", "label": "Int8"}, keep_default_na=False)
    entities = prepare_snapshot(clean, OUTPUT_DIR, REPO_PATH)
    if args.prepare_only:
        print(f"Prepared Feast snapshot: {len(entities):,} URL records.")
        return
    store = register_features()
    for split in ("train", "validation", "test"):
        historical = retrieve_historical(store, entities, split)
        saved = pd.read_csv(
            OUTPUT_DIR / f"{split}.csv",
            dtype={**dict.fromkeys(FEATURE_COLUMNS, "int32"), "label": "int8"},
        )
        if not historical.equals(saved):
            raise ValueError(f"Feast historical retrieval differs from Phase 3 {split}.")
        print(f"Historical {split}: {len(historical):,} rows, exact feature and label match")

    if not args.skip_online:
        materialize_online(store)
        sample_url = clean["url"].iloc[0]
        online = retrieve_online(store, sample_url)
        if online != extract_url_features(sample_url):
            raise ValueError("Online features differ from the shared URL extractor.")
        print(f"Online SQLite: materialized {len(entities):,} rows; sample matches the shared extractor")
        print(f"Sample feature vector: {online}")


if __name__ == "__main__":
    main()
