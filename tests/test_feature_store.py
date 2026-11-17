from shutil import copyfile

import pandas as pd
import pytest
from feast.types import Int32

from feature_store.feature_repo.entities import URL_ID, url_record_id
from feature_store.feature_repo.feature_views import FEATURE_REFS, VIEW_NAME
from feature_store.feature_repo.features import EVENT_TIMESTAMP, FEATURE_FILE
from feature_store.materialize import (
    ENTITY_FILE,
    REPO_PATH,
    SNAPSHOT_TIME,
    materialize_online,
    prepare_snapshot,
    register_features,
    retrieve_historical,
    retrieve_online,
)
from src.features.preprocessing import FEATURE_COLUMNS, extract_url_features, preprocess_dataset


@pytest.fixture(scope="module")
def local_store(tmp_path_factory):
    root = tmp_path_factory.mktemp("feast")
    copyfile(REPO_PATH / "feature_store.yaml", root / "feature_store.yaml")
    clean = pd.DataFrame(
        [
            (f"https://{host}.site{group:02}.co.uk/{group}", label)
            for group in range(20)
            for host, label in (("login", 0), ("shop", 1))
        ],
        columns=["url", "label"],
    )
    splits, _ = preprocess_dataset(clean)
    for name, frame in splits.items():
        frame.to_csv(root / f"{name}.csv", index=False)
    entities = prepare_snapshot(clean, root, root)
    return root, register_features(root), clean, splits, entities


def test_registered_schema_and_snapshot(local_store):
    root, store, clean, _, entities = local_store
    view = store.get_feature_view(VIEW_NAME)
    assert [(feature.name, feature.dtype) for feature in view.features] == [
        (name, Int32) for name in FEATURE_COLUMNS
    ]
    source = pd.read_parquet(root / "data" / FEATURE_FILE)
    assert list(source) == [URL_ID, EVENT_TIMESTAMP, *FEATURE_COLUMNS]
    assert source[URL_ID].is_unique
    assert set(source[URL_ID]) == set(clean["url"].map(url_record_id))
    assert source[EVENT_TIMESTAMP].eq(SNAPSHOT_TIME).all()
    assert url_record_id(" https://example.test ") == url_record_id("https://example.test")
    assert not source.isna().any().any()
    assert all(str(source[name].dtype) == "int32" for name in FEATURE_COLUMNS)
    pd.testing.assert_frame_equal(pd.read_parquet(root / "data" / ENTITY_FILE), entities)


def test_historical_features_match_every_saved_split(local_store):
    _, store, _, splits, entities = local_store
    for name, expected in splits.items():
        pd.testing.assert_frame_equal(retrieve_historical(store, entities, name), expected)


def test_historical_features_exclude_future_rows_and_fail_on_missing(local_store):
    _, store, _, _, entities = local_store
    request = entities.loc[entities["split"].eq("train")].head(1).copy()
    request[EVENT_TIMESTAMP] -= pd.Timedelta(seconds=1)
    result = store.get_historical_features(entity_df=request, features=FEATURE_REFS).to_df()
    assert result[FEATURE_COLUMNS].isna().all().all()
    with pytest.raises(ValueError, match="missing features"):
        retrieve_historical(store, request)


def test_online_features_match_shared_extractor_and_report_misses(local_store):
    _, store, clean, _, _ = local_store
    materialize_online(store)
    for url in clean["url"].iloc[[0, 1, -1]]:
        assert retrieve_online(store, url) == extract_url_features(url)
    with pytest.raises(LookupError, match="No complete online feature"):
        retrieve_online(store, "https://missing.example.test/")
