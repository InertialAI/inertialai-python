from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ._base_client import AsyncHTTPClient, SyncHTTPClient


class SyncAPIResource:
    _client: SyncHTTPClient

    def __init__(self, client: SyncHTTPClient) -> None:
        self._client = client


class AsyncAPIResource:
    _client: AsyncHTTPClient

    def __init__(self, client: AsyncHTTPClient) -> None:
        self._client = client
