"""Group the lexical features in an online-enabled view."""

from datetime import timedelta
from pathlib import Path

from feast import FeatureView

from feature_store.feature_repo.entities import url_record
from feature_store.feature_repo.features import FEATURE_FILE, make_url_source, url_feature_schema
from src.features.preprocessing import FEATURE_COLUMNS


VIEW_NAME = "url_lexical_features"
FEATURE_REFS = [f"{VIEW_NAME}:{name}" for name in FEATURE_COLUMNS]


def make_url_feature_view(path: Path) -> FeatureView:
    return FeatureView(
        name=VIEW_NAME,
        entities=[url_record],
        ttl=timedelta(0),
        schema=url_feature_schema,
        source=make_url_source(path),
        online=True,
    )


url_lexical_features = make_url_feature_view(Path(__file__).parent / "data" / FEATURE_FILE)
