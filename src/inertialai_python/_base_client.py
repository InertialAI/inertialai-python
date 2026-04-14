from __future__ import annotations

import asyncio
import random
import time
from typing import TypeVar

import httpx
from pydantic import BaseModel

from ._constants import (
    DEFAULT_BASE_URL,
    DEFAULT_MAX_RETRIES,
    DEFAULT_TIMEOUT,
    INITIAL_RETRY_DELAY,
    MAX_RETRY_DELAY,
    RETRYABLE_STATUS_CODES,
)
from ._exceptions import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    AuthenticationError,
    InternalServerError,
    RateLimitError,
    ValidationError,
)
from ._version import __version__

_T = TypeVar("_T", bound=BaseModel)


def _calculate_retry_delay(attempt: int, response: httpx.Response | None = None) -> float:
    if response is not None:
        retry_after = response.headers.get("retry-after")
        if retry_after is not None:
            try:
                return float(retry_after)
            except ValueError:
                pass

    delay: float = min(INITIAL_RETRY_DELAY * (2**attempt), MAX_RETRY_DELAY)
    jitter: float = random.uniform(0.75, 1.25)  # noqa: S311  # nosec B311
    return delay * jitter


def _extract_error_message(response: httpx.Response) -> str:
    try:
        body = response.json()
    except Exception:
        return response.text or f"HTTP {response.status_code}"

    if isinstance(body, dict):
        detail = body.get("detail")
        if isinstance(detail, str):
            return detail
        if isinstance(detail, list):
            messages = [item.get("msg", str(item)) for item in detail if isinstance(item, dict)]
            if messages:
                return "; ".join(messages)

    return response.text or f"HTTP {response.status_code}"


def _make_status_error(response: httpx.Response) -> APIStatusError:
    message = _extract_error_message(response)
    body: object = None
    try:
        body = response.json()
    except Exception:
        body = response.text

    error_class: type[APIStatusError]
    if response.status_code == 401:
        error_class = AuthenticationError
    elif response.status_code == 422:
        error_class = ValidationError
    elif response.status_code == 429:
        error_class = RateLimitError
    elif response.status_code >= 500:
        error_class = InternalServerError
    else:
        error_class = APIStatusError

    return error_class(message, response=response, body=body)


def _is_retryable(status_code: int) -> bool:
    return status_code in RETRYABLE_STATUS_CODES


def _default_headers(api_key: str) -> dict[str, str]:
    return {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "User-Agent": f"inertialai-python/{__version__}",
    }


class SyncHTTPClient:
    _client: httpx.Client
    _base_url: str
    _api_key: str
    _max_retries: int
    _owns_client: bool

    def __init__(
        self,
        *,
        api_key: str,
        base_url: str = DEFAULT_BASE_URL,
        timeout: float | httpx.Timeout | None = None,
        max_retries: int = DEFAULT_MAX_RETRIES,
        http_client: httpx.Client | None = None,
    ) -> None:
        self._api_key = api_key
        self._base_url = base_url.rstrip("/")
        self._max_retries = max_retries

        if isinstance(timeout, (int, float)):
            resolved_timeout = httpx.Timeout(timeout)
        else:
            resolved_timeout = timeout or DEFAULT_TIMEOUT

        if http_client is not None:
            self._client = http_client
            self._owns_client = False
        else:
            self._client = httpx.Client(timeout=resolved_timeout)
            self._owns_client = True

    def _request(
        self,
        method: str,
        path: str,
        *,
        body: dict[str, object] | None = None,
        cast_to: type[_T],
        timeout: float | httpx.Timeout | None = None,
    ) -> _T:
        url = f"{self._base_url}{path}"
        headers = _default_headers(self._api_key)
        request_timeout = httpx.Timeout(timeout) if isinstance(timeout, (int, float)) else timeout

        last_error: Exception | None = None
        for attempt in range(self._max_retries + 1):
            try:
                response = self._client.request(
                    method,
                    url,
                    headers=headers,
                    json=body,
                    timeout=request_timeout,
                )
            except httpx.TimeoutException as e:
                last_error = APITimeoutError(request=getattr(e, "request", None))
                if attempt < self._max_retries:
                    time.sleep(_calculate_retry_delay(attempt))
                    continue
                raise last_error from e
            except httpx.ConnectError as e:
                last_error = APIConnectionError(request=getattr(e, "request", None))
                if attempt < self._max_retries:
                    time.sleep(_calculate_retry_delay(attempt))
                    continue
                raise last_error from e

            if response.is_success:
                return cast_to.model_validate(response.json())

            if _is_retryable(response.status_code) and attempt < self._max_retries:
                time.sleep(_calculate_retry_delay(attempt, response))
                continue

            raise _make_status_error(response)

        # Should not be reached, but satisfies type checker
        raise last_error or APIConnectionError("Max retries exceeded")  # pragma: no cover

    def post(
        self,
        path: str,
        *,
        body: dict[str, object],
        cast_to: type[_T],
        timeout: float | httpx.Timeout | None = None,
    ) -> _T:
        return self._request("POST", path, body=body, cast_to=cast_to, timeout=timeout)

    def close(self) -> None:
        if self._owns_client:
            self._client.close()

    def __enter__(self) -> SyncHTTPClient:
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()


class AsyncHTTPClient:
    _client: httpx.AsyncClient
    _base_url: str
    _api_key: str
    _max_retries: int
    _owns_client: bool

    def __init__(
        self,
        *,
        api_key: str,
        base_url: str = DEFAULT_BASE_URL,
        timeout: float | httpx.Timeout | None = None,
        max_retries: int = DEFAULT_MAX_RETRIES,
        http_client: httpx.AsyncClient | None = None,
    ) -> None:
        self._api_key = api_key
        self._base_url = base_url.rstrip("/")
        self._max_retries = max_retries

        if isinstance(timeout, (int, float)):
            resolved_timeout = httpx.Timeout(timeout)
        else:
            resolved_timeout = timeout or DEFAULT_TIMEOUT

        if http_client is not None:
            self._client = http_client
            self._owns_client = False
        else:
            self._client = httpx.AsyncClient(timeout=resolved_timeout)
            self._owns_client = True

    async def _request(
        self,
        method: str,
        path: str,
        *,
        body: dict[str, object] | None = None,
        cast_to: type[_T],
        timeout: float | httpx.Timeout | None = None,
    ) -> _T:
        url = f"{self._base_url}{path}"
        headers = _default_headers(self._api_key)
        request_timeout = httpx.Timeout(timeout) if isinstance(timeout, (int, float)) else timeout

        last_error: Exception | None = None
        for attempt in range(self._max_retries + 1):
            try:
                response = await self._client.request(
                    method,
                    url,
                    headers=headers,
                    json=body,
                    timeout=request_timeout,
                )
            except httpx.TimeoutException as e:
                last_error = APITimeoutError(request=getattr(e, "request", None))
                if attempt < self._max_retries:
                    await asyncio.sleep(_calculate_retry_delay(attempt))
                    continue
                raise last_error from e
            except httpx.ConnectError as e:
                last_error = APIConnectionError(request=getattr(e, "request", None))
                if attempt < self._max_retries:
                    await asyncio.sleep(_calculate_retry_delay(attempt))
                    continue
                raise last_error from e

            if response.is_success:
                return cast_to.model_validate(response.json())

            if _is_retryable(response.status_code) and attempt < self._max_retries:
                await asyncio.sleep(_calculate_retry_delay(attempt, response))
                continue

            raise _make_status_error(response)

        raise last_error or APIConnectionError("Max retries exceeded")  # pragma: no cover

    async def post(
        self,
        path: str,
        *,
        body: dict[str, object],
        cast_to: type[_T],
        timeout: float | httpx.Timeout | None = None,
    ) -> _T:
        return await self._request("POST", path, body=body, cast_to=cast_to, timeout=timeout)

    async def close(self) -> None:
        if self._owns_client:
            await self._client.aclose()

    async def __aenter__(self) -> AsyncHTTPClient:
        return self

    async def __aexit__(self, *_args: object) -> None:
        await self.close()
