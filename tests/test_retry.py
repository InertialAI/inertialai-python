from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import httpx
import pytest
import respx

from inertialai_python import (
    APIConnectionError,
    APITimeoutError,
    AsyncInertialAI,
    AuthenticationError,
    EmbeddingModel,
    InertialAI,
    InternalServerError,
    RateLimitError,
)
from inertialai_python.types import EmbeddingMultiModalInput

from .conftest import TEST_BASE_URL


class TestRetrySync:
    @respx.mock
    @patch("time.sleep")
    def test_retries_on_500(
        self,
        mock_sleep: MagicMock,
        sample_embedding_response: dict[str, Any],
    ) -> None:
        client = InertialAI(api_key="iai_test", base_url=TEST_BASE_URL, max_retries=2)
        route = respx.post(f"{TEST_BASE_URL}/api/v1/embeddings").mock(
            side_effect=[
                httpx.Response(500, json={"detail": "error"}),
                httpx.Response(200, json=sample_embedding_response),
            ]
        )
        result = client.embeddings.create(
            input=[EmbeddingMultiModalInput(text="test")],
            model=EmbeddingModel.INERTIAL_EMBED_ALPHA,
        )
        assert route.call_count == 2
        assert result.model == EmbeddingModel.INERTIAL_EMBED_ALPHA
        assert mock_sleep.called
        client.close()

    @respx.mock
    @patch("time.sleep")
    def test_retries_on_429(
        self,
        mock_sleep: MagicMock,
        sample_embedding_response: dict[str, Any],
    ) -> None:
        client = InertialAI(api_key="iai_test", base_url=TEST_BASE_URL, max_retries=1)
        route = respx.post(f"{TEST_BASE_URL}/api/v1/embeddings").mock(
            side_effect=[
                httpx.Response(429, json={"detail": "rate limited"}),
                httpx.Response(200, json=sample_embedding_response),
            ]
        )
        result = client.embeddings.create(
            input=[EmbeddingMultiModalInput(text="test")],
            model=EmbeddingModel.INERTIAL_EMBED_ALPHA,
        )
        assert route.call_count == 2
        assert isinstance(result.data, list)
        client.close()

    @respx.mock
    @patch("time.sleep")
    def test_exhausts_retries_then_raises(self, mock_sleep: MagicMock) -> None:
        client = InertialAI(api_key="iai_test", base_url=TEST_BASE_URL, max_retries=2)
        respx.post(f"{TEST_BASE_URL}/api/v1/embeddings").mock(
            return_value=httpx.Response(500, json={"detail": "error"})
        )
        with pytest.raises(InternalServerError):
            client.embeddings.create(
                input=[EmbeddingMultiModalInput(text="test")],
                model=EmbeddingModel.INERTIAL_EMBED_ALPHA,
            )
        client.close()

    @respx.mock
    @patch("time.sleep")
    def test_no_retry_on_401(self, mock_sleep: MagicMock) -> None:
        client = InertialAI(api_key="iai_test", base_url=TEST_BASE_URL, max_retries=2)
        route = respx.post(f"{TEST_BASE_URL}/api/v1/embeddings").mock(
            return_value=httpx.Response(401, json={"detail": "unauthorized"})
        )
        with pytest.raises(AuthenticationError):
            client.embeddings.create(
                input=[EmbeddingMultiModalInput(text="test")],
                model=EmbeddingModel.INERTIAL_EMBED_ALPHA,
            )
        assert route.call_count == 1
        client.close()

    @respx.mock
    @patch("time.sleep")
    def test_retries_on_timeout(
        self,
        mock_sleep: MagicMock,
        sample_embedding_response: dict[str, Any],
    ) -> None:
        client = InertialAI(api_key="iai_test", base_url=TEST_BASE_URL, max_retries=1)
        route = respx.post(f"{TEST_BASE_URL}/api/v1/embeddings").mock(
            side_effect=[
                httpx.ReadTimeout("timeout"),
                httpx.Response(200, json=sample_embedding_response),
            ]
        )
        result = client.embeddings.create(
            input=[EmbeddingMultiModalInput(text="test")],
            model=EmbeddingModel.INERTIAL_EMBED_ALPHA,
        )
        assert route.call_count == 2
        assert isinstance(result, type(result))
        client.close()

    @respx.mock
    @patch("time.sleep")
    def test_timeout_exhausts_retries(self, mock_sleep: MagicMock) -> None:
        client = InertialAI(api_key="iai_test", base_url=TEST_BASE_URL, max_retries=1)
        respx.post(f"{TEST_BASE_URL}/api/v1/embeddings").mock(
            side_effect=httpx.ReadTimeout("timeout")
        )
        with pytest.raises(APITimeoutError):
            client.embeddings.create(
                input=[EmbeddingMultiModalInput(text="test")],
                model=EmbeddingModel.INERTIAL_EMBED_ALPHA,
            )
        client.close()

    @respx.mock
    @patch("time.sleep")
    def test_connection_error_retries(self, mock_sleep: MagicMock) -> None:
        client = InertialAI(api_key="iai_test", base_url=TEST_BASE_URL, max_retries=1)
        respx.post(f"{TEST_BASE_URL}/api/v1/embeddings").mock(
            side_effect=httpx.ConnectError("connection failed")
        )
        with pytest.raises(APIConnectionError):
            client.embeddings.create(
                input=[EmbeddingMultiModalInput(text="test")],
                model=EmbeddingModel.INERTIAL_EMBED_ALPHA,
            )
        client.close()

    @respx.mock
    @patch("time.sleep")
    def test_respects_retry_after_header(
        self,
        mock_sleep: MagicMock,
        sample_embedding_response: dict[str, Any],
    ) -> None:
        client = InertialAI(api_key="iai_test", base_url=TEST_BASE_URL, max_retries=1)
        respx.post(f"{TEST_BASE_URL}/api/v1/embeddings").mock(
            side_effect=[
                httpx.Response(
                    429,
                    json={"detail": "rate limited"},
                    headers={"retry-after": "2"},
                ),
                httpx.Response(200, json=sample_embedding_response),
            ]
        )
        client.embeddings.create(
            input=[EmbeddingMultiModalInput(text="test")],
            model=EmbeddingModel.INERTIAL_EMBED_ALPHA,
        )
        mock_sleep.assert_called_once_with(2.0)
        client.close()

    @respx.mock
    def test_no_retry_when_max_retries_zero(self) -> None:
        client = InertialAI(api_key="iai_test", base_url=TEST_BASE_URL, max_retries=0)
        route = respx.post(f"{TEST_BASE_URL}/api/v1/embeddings").mock(
            return_value=httpx.Response(500, json={"detail": "error"})
        )
        with pytest.raises(InternalServerError):
            client.embeddings.create(
                input=[EmbeddingMultiModalInput(text="test")],
                model=EmbeddingModel.INERTIAL_EMBED_ALPHA,
            )
        assert route.call_count == 1
        client.close()


class TestRetryAsync:
    @respx.mock
    @pytest.mark.asyncio
    async def test_retries_on_500(self, sample_embedding_response: dict[str, Any]) -> None:
        client = AsyncInertialAI(api_key="iai_test", base_url=TEST_BASE_URL, max_retries=1)
        route = respx.post(f"{TEST_BASE_URL}/api/v1/embeddings").mock(
            side_effect=[
                httpx.Response(500, json={"detail": "error"}),
                httpx.Response(200, json=sample_embedding_response),
            ]
        )
        with patch("asyncio.sleep"):
            result = await client.embeddings.create(
                input=[EmbeddingMultiModalInput(text="test")],
                model=EmbeddingModel.INERTIAL_EMBED_ALPHA,
            )
        assert route.call_count == 2
        assert isinstance(result, type(result))
        await client.close()

    @respx.mock
    @pytest.mark.asyncio
    async def test_exhausts_retries_raises(self) -> None:
        client = AsyncInertialAI(api_key="iai_test", base_url=TEST_BASE_URL, max_retries=1)
        respx.post(f"{TEST_BASE_URL}/api/v1/embeddings").mock(
            return_value=httpx.Response(429, json={"detail": "rate limited"})
        )
        with patch("asyncio.sleep"), pytest.raises(RateLimitError):
            await client.embeddings.create(
                input=[EmbeddingMultiModalInput(text="test")],
                model=EmbeddingModel.INERTIAL_EMBED_ALPHA,
            )
        await client.close()
