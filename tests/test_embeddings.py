from __future__ import annotations

from typing import Any

import httpx
import pytest
import respx

from inertialai_python import (
    AsyncInertialAI,
    AuthenticationError,
    EmbeddingModel,
    EmbeddingResponse,
    InertialAI,
    InternalServerError,
    RateLimitError,
    ValidationError,
)
from inertialai_python.types import EmbeddingMultiModalInput

from .conftest import TEST_BASE_URL


class TestEmbeddingsCreate:
    @respx.mock
    def test_create_with_time_series(
        self, sync_client: InertialAI, sample_embedding_response: dict[str, Any]
    ) -> None:
        route = respx.post(f"{TEST_BASE_URL}/api/v1/embeddings").mock(
            return_value=httpx.Response(200, json=sample_embedding_response)
        )
        result = sync_client.embeddings.create(
            input=[EmbeddingMultiModalInput(time_series=[[1.0, 2.0, 3.0]])],
            model=EmbeddingModel.INERTIAL_EMBED_ALPHA,
        )
        assert isinstance(result, EmbeddingResponse)
        assert result.model == EmbeddingModel.INERTIAL_EMBED_ALPHA
        assert len(result.data) == 1
        assert result.data[0].embedding == [0.1, 0.2, 0.3]
        assert result.usage.prompt_tokens == 10
        assert route.called

    @respx.mock
    def test_create_with_text(
        self, sync_client: InertialAI, sample_embedding_response: dict[str, Any]
    ) -> None:
        route = respx.post(f"{TEST_BASE_URL}/api/v1/embeddings").mock(
            return_value=httpx.Response(200, json=sample_embedding_response)
        )
        result = sync_client.embeddings.create(
            input=[EmbeddingMultiModalInput(text="sensor reading")],
            model=EmbeddingModel.INERTIAL_EMBED_ALPHA,
        )
        assert isinstance(result, EmbeddingResponse)
        assert route.called

    @respx.mock
    def test_create_with_dict_input(
        self, sync_client: InertialAI, sample_embedding_response: dict[str, Any]
    ) -> None:
        route = respx.post(f"{TEST_BASE_URL}/api/v1/embeddings").mock(
            return_value=httpx.Response(200, json=sample_embedding_response)
        )
        result = sync_client.embeddings.create(
            input=[{"time_series": [[1.0, 2.0]], "text": "test"}],  # type: ignore[list-item]
            model=EmbeddingModel.INERTIAL_EMBED_ALPHA,
        )
        assert isinstance(result, EmbeddingResponse)
        assert route.called

    @respx.mock
    def test_create_with_optional_params(
        self, sync_client: InertialAI, sample_embedding_response: dict[str, Any]
    ) -> None:
        route = respx.post(f"{TEST_BASE_URL}/api/v1/embeddings").mock(
            return_value=httpx.Response(200, json=sample_embedding_response)
        )
        from inertialai_python import EmbeddingEncodingFormat

        result = sync_client.embeddings.create(
            input=[EmbeddingMultiModalInput(text="test")],
            model=EmbeddingModel.INERTIAL_EMBED_ALPHA,
            dimensions=512,
            encoding_format=EmbeddingEncodingFormat.FLOAT,
        )
        assert isinstance(result, EmbeddingResponse)

        request_body = route.calls[0].request
        import json

        body = json.loads(request_body.content)
        assert body["dimensions"] == 512
        assert body["encoding_format"] == "float"

    @respx.mock
    def test_401_raises_authentication_error(self, sync_client: InertialAI) -> None:
        respx.post(f"{TEST_BASE_URL}/api/v1/embeddings").mock(
            return_value=httpx.Response(401, json={"detail": "Invalid API key"})
        )
        with pytest.raises(AuthenticationError, match="Invalid API key"):
            sync_client.embeddings.create(
                input=[EmbeddingMultiModalInput(text="test")],
                model=EmbeddingModel.INERTIAL_EMBED_ALPHA,
            )

    @respx.mock
    def test_422_raises_validation_error(self, sync_client: InertialAI) -> None:
        respx.post(f"{TEST_BASE_URL}/api/v1/embeddings").mock(
            return_value=httpx.Response(
                422, json={"detail": [{"msg": "field required", "type": "missing"}]}
            )
        )
        with pytest.raises(ValidationError, match="field required"):
            sync_client.embeddings.create(
                input=[EmbeddingMultiModalInput(text="test")],
                model=EmbeddingModel.INERTIAL_EMBED_ALPHA,
            )

    @respx.mock
    def test_429_raises_rate_limit_error(self, sync_client: InertialAI) -> None:
        client = InertialAI(api_key="iai_test", base_url=TEST_BASE_URL, max_retries=0)
        respx.post(f"{TEST_BASE_URL}/api/v1/embeddings").mock(
            return_value=httpx.Response(429, json={"detail": "Too many requests"})
        )
        with pytest.raises(RateLimitError):
            client.embeddings.create(
                input=[EmbeddingMultiModalInput(text="test")],
                model=EmbeddingModel.INERTIAL_EMBED_ALPHA,
            )
        client.close()

    @respx.mock
    def test_500_raises_internal_server_error(self, sync_client: InertialAI) -> None:
        client = InertialAI(api_key="iai_test", base_url=TEST_BASE_URL, max_retries=0)
        respx.post(f"{TEST_BASE_URL}/api/v1/embeddings").mock(
            return_value=httpx.Response(500, json={"detail": "Internal error"})
        )
        with pytest.raises(InternalServerError):
            client.embeddings.create(
                input=[EmbeddingMultiModalInput(text="test")],
                model=EmbeddingModel.INERTIAL_EMBED_ALPHA,
            )
        client.close()


class TestAsyncEmbeddingsCreate:
    @respx.mock
    @pytest.mark.asyncio
    async def test_create(
        self, async_client: AsyncInertialAI, sample_embedding_response: dict[str, Any]
    ) -> None:
        route = respx.post(f"{TEST_BASE_URL}/api/v1/embeddings").mock(
            return_value=httpx.Response(200, json=sample_embedding_response)
        )
        result = await async_client.embeddings.create(
            input=[EmbeddingMultiModalInput(time_series=[[1.0, 2.0, 3.0]])],
            model=EmbeddingModel.INERTIAL_EMBED_ALPHA,
        )
        assert isinstance(result, EmbeddingResponse)
        assert route.called

    @respx.mock
    @pytest.mark.asyncio
    async def test_401_raises_authentication_error(self, async_client: AsyncInertialAI) -> None:
        respx.post(f"{TEST_BASE_URL}/api/v1/embeddings").mock(
            return_value=httpx.Response(401, json={"detail": "Unauthorized"})
        )
        with pytest.raises(AuthenticationError):
            await async_client.embeddings.create(
                input=[EmbeddingMultiModalInput(text="test")],
                model=EmbeddingModel.INERTIAL_EMBED_ALPHA,
            )
