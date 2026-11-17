"""Stable URL record identity for offline and online lookup."""

from hashlib import sha256

from feast import Entity, ValueType


URL_ID = "url_id"
url_record = Entity(
    name="url_record",
    join_keys=[URL_ID],
    value_type=ValueType.STRING,
    description="SHA-256 of the trimmed URL encoded as UTF-8.",
)


def url_record_id(url: str) -> str:
    return sha256(url.strip().encode("utf-8")).hexdigest()
