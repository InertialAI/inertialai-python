from __future__ import annotations

import functools
import os

import httpx

from ._base_client import AsyncHTTPClient, SyncHTTPClient
from ._constants import DEFAULT_BASE_URL, DEFAULT_MAX_RETRIES
from ._exceptions import InertialAIError
from .resources.embeddings import AsyncEmbeddings, Embeddings


class InertialAI(SyncHTTPClient):
    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        timeout: float | httpx.Timeout | None = None,
        max_retries: int = DEFAULT_MAX_RETRIES,
        http_client: httpx.Client | None = None,
    ) -> None:
        resolved_api_key = api_key or os.environ.get("INERTIALAI_API_KEY")
        if not resolved_api_key:
            raise InertialAIError(
                "The api_key client option must be set either by passing api_key to the client "
                "or by setting the INERTIALAI_API_KEY environment variable"
            )
        super().__init__(
            api_key=resolved_api_key,
            base_url=base_url or DEFAULT_BASE_URL,
            timeout=timeout,
            max_retries=max_retries,
            http_client=http_client,
        )

    @functools.cached_property
    def embeddings(self) -> Embeddings:
        return Embeddings(self)

    def __enter__(self) -> InertialAI:
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()


class AsyncInertialAI(AsyncHTTPClient):
    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        timeout: float | httpx.Timeout | None = None,
        max_retries: int = DEFAULT_MAX_RETRIES,
        http_client: httpx.AsyncClient | None = None,
    ) -> None:
        resolved_api_key = api_key or os.environ.get("INERTIALAI_API_KEY")
        if not resolved_api_key:
            raise InertialAIError(
                "The api_key client option must be set either by passing api_key to the client "
                "or by setting the INERTIALAI_API_KEY environment variable"
            )
        super().__init__(
            api_key=resolved_api_key,
            base_url=base_url or DEFAULT_BASE_URL,
            timeout=timeout,
            max_retries=max_retries,
            http_client=http_client,
        )

    @functools.cached_property
    def embeddings(self) -> AsyncEmbeddings:
        return AsyncEmbeddings(self)

    async def __aenter__(self) -> AsyncInertialAI:
        return self

    async def __aexit__(self, *_args: object) -> None:
        await self.close()
