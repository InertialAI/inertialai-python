from __future__ import annotations

import pytest
from pydantic import ValidationError

from inertialai_python.types import (
    CreateEmbeddingRequest,
    EmbeddingData,
    EmbeddingEncodingFormat,
    EmbeddingModel,
    EmbeddingMultiModalInput,
    EmbeddingResponse,
    EmbeddingUsage,
)


class TestEmbeddingMultiModalInput:
    def test_time_series_only(self) -> None:
        inp = EmbeddingMultiModalInput(time_series=[[1.0, 2.0, 3.0]])
        assert inp.time_series == [[1.0, 2.0, 3.0]]
        assert inp.text is None

    def test_text_only(self) -> None:
        inp = EmbeddingMultiModalInput(text="sensor reading")
        assert inp.text == "sensor reading"
        assert inp.time_series is None

    def test_both_fields(self) -> None:
        inp = EmbeddingMultiModalInput(time_series=[[1.0]], text="test")
        assert inp.time_series == [[1.0]]
        assert inp.text == "test"

    def test_neither_field_raises(self) -> None:
        with pytest.raises(ValidationError, match="At least one"):
            EmbeddingMultiModalInput()

    def test_from_dict(self) -> None:
        inp = EmbeddingMultiModalInput.model_validate({"text": "hello"})
        assert inp.text == "hello"


class TestCreateEmbeddingRequest:
    def test_valid_request(self) -> None:
        req = CreateEmbeddingRequest(
            input=[EmbeddingMultiModalInput(text="test")],
            model=EmbeddingModel.INERTIAL_EMBED_ALPHA,
        )
        assert req.model == EmbeddingModel.INERTIAL_EMBED_ALPHA
        assert req.dimensions is None
        assert req.encoding_format is None

    def test_with_optional_fields(self) -> None:
        req = CreateEmbeddingRequest(
            input=[EmbeddingMultiModalInput(text="test")],
            model=EmbeddingModel.INERTIAL_EMBED_ALPHA,
            dimensions=512,
            encoding_format=EmbeddingEncodingFormat.BASE64,
        )
        assert req.dimensions == 512
        assert req.encoding_format == EmbeddingEncodingFormat.BASE64

    def test_model_dump_excludes_none(self) -> None:
        req = CreateEmbeddingRequest(
            input=[EmbeddingMultiModalInput(text="test")],
            model=EmbeddingModel.INERTIAL_EMBED_ALPHA,
        )
        dumped = req.model_dump(exclude_none=True)
        assert "dimensions" not in dumped
        assert "encoding_format" not in dumped
        assert "model" in dumped
        assert "input" in dumped


class TestEmbeddingModel:
    def test_enum_value(self) -> None:
        assert EmbeddingModel.INERTIAL_EMBED_ALPHA == "inertial-embed-alpha"
        assert EmbeddingModel.INERTIAL_EMBED_ALPHA.value == "inertial-embed-alpha"


class TestEmbeddingEncodingFormat:
    def test_enum_values(self) -> None:
        assert EmbeddingEncodingFormat.FLOAT == "float"
        assert EmbeddingEncodingFormat.BASE64 == "base64"


class TestEmbeddingResponse:
    def test_valid_response(self) -> None:
        resp = EmbeddingResponse.model_validate(
            {
                "id": "emb_123",
                "object": "list",
                "model": "inertial-embed-alpha",
                "data": [{"object": "embedding", "index": 0, "embedding": [0.1, 0.2]}],
                "usage": {"prompt_tokens": 5, "total_tokens": 5},
                "create_time": "2024-01-01T00:00:00Z",
            }
        )
        assert resp.id == "emb_123"
        assert resp.model == EmbeddingModel.INERTIAL_EMBED_ALPHA
        assert len(resp.data) == 1

    def test_extra_fields_ignored(self) -> None:
        resp = EmbeddingResponse.model_validate(
            {
                "id": "emb_123",
                "object": "list",
                "model": "inertial-embed-alpha",
                "data": [{"object": "embedding", "index": 0, "embedding": [0.1], "extra": True}],
                "usage": {"prompt_tokens": 5, "total_tokens": 5, "extra": True},
                "create_time": "2024-01-01T00:00:00Z",
                "unknown_field": "ignored",
            }
        )
        assert resp.id == "emb_123"

    def test_id_optional(self) -> None:
        resp = EmbeddingResponse.model_validate(
            {
                "object": "list",
                "model": "inertial-embed-alpha",
                "data": [{"object": "embedding", "index": 0, "embedding": [0.1]}],
                "usage": {"prompt_tokens": 5, "total_tokens": 5},
                "create_time": "2024-01-01T00:00:00Z",
            }
        )
        assert resp.id is None

    def test_base64_embedding(self) -> None:
        data = EmbeddingData.model_validate(
            {"object": "embedding", "index": 0, "embedding": "AAAA"}
        )
        assert data.embedding == "AAAA"


class TestEmbeddingUsage:
    def test_valid_usage(self) -> None:
        usage = EmbeddingUsage(prompt_tokens=10, total_tokens=15)
        assert usage.prompt_tokens == 10
        assert usage.total_tokens == 15
