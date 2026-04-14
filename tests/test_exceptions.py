from __future__ import annotations

import httpx

from inertialai_python import (
    APIConnectionError,
    APIError,
    APIStatusError,
    APITimeoutError,
    AuthenticationError,
    InertialAIError,
    InternalServerError,
    RateLimitError,
    ValidationError,
)


class TestExceptionHierarchy:
    def test_base_exception(self) -> None:
        err = InertialAIError("test")
        assert isinstance(err, Exception)
        assert str(err) == "test"

    def test_api_error(self) -> None:
        err = APIError("msg", body={"detail": "test"})
        assert isinstance(err, InertialAIError)
        assert err.message == "msg"
        assert err.body == {"detail": "test"}
        assert err.request is None

    def test_api_status_error(self) -> None:
        response = httpx.Response(
            400,
            json={"detail": "bad request"},
            request=httpx.Request("POST", "https://example.com"),
        )
        err = APIStatusError("bad request", response=response)
        assert isinstance(err, APIError)
        assert err.status_code == 400
        assert err.response is response

    def test_authentication_error(self) -> None:
        response = httpx.Response(
            401,
            json={"detail": "unauthorized"},
            request=httpx.Request("POST", "https://example.com"),
        )
        err = AuthenticationError("unauthorized", response=response)
        assert isinstance(err, APIStatusError)
        assert err.status_code == 401

    def test_validation_error(self) -> None:
        response = httpx.Response(
            422,
            json={"detail": "invalid"},
            request=httpx.Request("POST", "https://example.com"),
        )
        err = ValidationError("invalid", response=response)
        assert isinstance(err, APIStatusError)

    def test_rate_limit_error(self) -> None:
        response = httpx.Response(
            429,
            json={"detail": "rate limited"},
            request=httpx.Request("POST", "https://example.com"),
        )
        err = RateLimitError("rate limited", response=response)
        assert isinstance(err, APIStatusError)

    def test_internal_server_error(self) -> None:
        response = httpx.Response(
            500,
            json={"detail": "server error"},
            request=httpx.Request("POST", "https://example.com"),
        )
        err = InternalServerError("server error", response=response)
        assert isinstance(err, APIStatusError)

    def test_connection_error(self) -> None:
        err = APIConnectionError()
        assert isinstance(err, APIError)
        assert "Connection error" in str(err)

    def test_timeout_error(self) -> None:
        err = APITimeoutError()
        assert isinstance(err, APIConnectionError)
        assert "timed out" in str(err)
