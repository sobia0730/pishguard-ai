"""Declare the Parquet source and ten shared Feast feature fields."""

from pathlib import Path

from feast import Field, FileSource
from feast.types import Int32

from src.features.preprocessing import FEATURE_DEFINITIONS


EVENT_TIMESTAMP = "event_timestamp"
FEATURE_FILE = "url_features.parquet"

url_feature_schema = [
    Field(name=name, dtype=Int32, description=definition)
    for name, definition in FEATURE_DEFINITIONS.items()
]


def make_url_source(path: Path) -> FileSource:
    return FileSource(
        name="url_feature_snapshot",
        path=path.resolve().as_posix(),
        timestamp_field=EVENT_TIMESTAMP,
    )
