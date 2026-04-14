from __future__ import annotations

import httpx

from .._resource import AsyncAPIResource, SyncAPIResource
from ..types import (
    CreateEmbeddingRequest,
    EmbeddingEncodingFormat,
    EmbeddingModel,
    EmbeddingMultiModalInput,
    EmbeddingResponse,
)


class Embeddings(SyncAPIResource):
    def create(
        self,
        *,
        input: list[EmbeddingMultiModalInput],
        model: EmbeddingModel,
        dimensions: int | None = None,
        encoding_format: EmbeddingEncodingFormat | None = None,
        timeout: float | httpx.Timeout | None = None,
    ) -> EmbeddingResponse:
        request = CreateEmbeddingRequest(
            input=input,
            model=model,
            dimensions=dimensions,
            encoding_format=encoding_format,
        )
        return self._client.post(
            "/api/v1/embeddings",
            body=request.model_dump(exclude_none=True),
            cast_to=EmbeddingResponse,
            timeout=timeout,
        )


class AsyncEmbeddings(AsyncAPIResource):
    async def create(
        self,
        *,
        input: list[EmbeddingMultiModalInput],
        model: EmbeddingModel,
        dimensions: int | None = None,
        encoding_format: EmbeddingEncodingFormat | None = None,
        timeout: float | httpx.Timeout | None = None,
    ) -> EmbeddingResponse:
        request = CreateEmbeddingRequest(
            input=input,
            model=model,
            dimensions=dimensions,
            encoding_format=encoding_format,
        )
        return await self._client.post(
            "/api/v1/embeddings",
            body=request.model_dump(exclude_none=True),
            cast_to=EmbeddingResponse,
            timeout=timeout,
        )
