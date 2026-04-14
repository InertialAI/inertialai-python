from __future__ import annotations

import httpx


class InertialAIError(Exception):
    pass


class APIError(InertialAIError):
    message: str
    request: httpx.Request | None
    body: object

    def __init__(
        self,
        message: str,
        *,
        request: httpx.Request | None = None,
        body: object = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.request = request
        self.body = body


class APIStatusError(APIError):
    response: httpx.Response
    status_code: int

    def __init__(
        self,
        message: str,
        *,
        response: httpx.Response,
        body: object = None,
    ) -> None:
        super().__init__(message, request=response.request, body=body)
        self.response = response
        self.status_code = response.status_code


class AuthenticationError(APIStatusError):
    pass


class ValidationError(APIStatusError):
    pass


class RateLimitError(APIStatusError):
    pass


class InternalServerError(APIStatusError):
    pass


class APIConnectionError(APIError):
    def __init__(
        self,
        message: str = "Connection error.",
        *,
        request: httpx.Request | None = None,
    ) -> None:
        super().__init__(message, request=request)


class APITimeoutError(APIConnectionError):
    def __init__(
        self,
        message: str = "Request timed out.",
        *,
        request: httpx.Request | None = None,
    ) -> None:
        super().__init__(message, request=request)
