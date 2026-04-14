from __future__ import annotations

import os
from unittest import mock

import pytest

from inertialai_python import AsyncInertialAI, InertialAI, InertialAIError
from inertialai_python.resources.embeddings import AsyncEmbeddings, Embeddings


class TestInertialAIClient:
    def test_explicit_api_key(self) -> None:
        client = InertialAI(api_key="iai_test")
        assert client._api_key == "iai_test"
        client.close()

    def test_env_var_api_key(self) -> None:
        with mock.patch.dict(os.environ, {"INERTIALAI_API_KEY": "iai_env"}):
            client = InertialAI()
            assert client._api_key == "iai_env"
            client.close()

    def test_explicit_takes_priority_over_env(self) -> None:
        with mock.patch.dict(os.environ, {"INERTIALAI_API_KEY": "iai_env"}):
            client = InertialAI(api_key="iai_explicit")
            assert client._api_key == "iai_explicit"
            client.close()

    def test_missing_api_key_raises(self) -> None:
        with (
            mock.patch.dict(os.environ, {}, clear=True),
            pytest.raises(InertialAIError, match="api_key"),
        ):
            InertialAI()

    def test_custom_base_url(self) -> None:
        client = InertialAI(api_key="iai_test", base_url="https://custom.example.com/")
        assert client._base_url == "https://custom.example.com"
        client.close()

    def test_embeddings_property(self, sync_client: InertialAI) -> None:
        assert isinstance(sync_client.embeddings, Embeddings)
        assert sync_client.embeddings is sync_client.embeddings

    def test_context_manager(self) -> None:
        with InertialAI(api_key="iai_test") as client:
            assert isinstance(client, InertialAI)


class TestAsyncInertialAIClient:
    def test_explicit_api_key(self) -> None:
        client = AsyncInertialAI(api_key="iai_test")
        assert client._api_key == "iai_test"

    def test_missing_api_key_raises(self) -> None:
        with (
            mock.patch.dict(os.environ, {}, clear=True),
            pytest.raises(InertialAIError, match="api_key"),
        ):
            AsyncInertialAI()

    def test_embeddings_property(self, async_client: AsyncInertialAI) -> None:
        assert isinstance(async_client.embeddings, AsyncEmbeddings)
        assert async_client.embeddings is async_client.embeddings

    @pytest.mark.asyncio
    async def test_async_context_manager(self) -> None:
        async with AsyncInertialAI(api_key="iai_test") as client:
            assert isinstance(client, AsyncInertialAI)
