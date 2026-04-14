from __future__ import annotations

from enum import StrEnum


class EmbeddingModel(StrEnum):
    INERTIAL_EMBED_ALPHA = "inertial-embed-alpha"


class EmbeddingEncodingFormat(StrEnum):
    FLOAT = "float"
    BASE64 = "base64"
