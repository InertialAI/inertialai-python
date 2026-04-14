from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import pytest

from inertialai_python import AsyncInertialAI, InertialAI

TEST_API_KEY = "iai_test_key_12345"
TEST_BASE_URL = "https://test.inertialai.com"


@pytest.fixture
def api_key() -> str:
    return TEST_API_KEY


@pytest.fixture
def base_url() -> str:
    return TEST_BASE_URL


@pytest.fixture
def sync_client(api_key: str, base_url: str) -> InertialAI:
    client = InertialAI(api_key=api_key, base_url=base_url)
    yield client  # type: ignore[misc]
    client.close()


@pytest.fixture
def async_client(api_key: str, base_url: str) -> AsyncInertialAI:
    return AsyncInertialAI(api_key=api_key, base_url=base_url)


@pytest.fixture
def sample_embedding_response() -> dict[str, Any]:
    return {
        "id": "emb_abc123",
        "object": "list",
        "model": "inertial-embed-alpha",
        "data": [
            {
                "object": "embedding",
                "index": 0,
                "embedding": [0.1, 0.2, 0.3],
            }
        ],
        "usage": {
            "prompt_tokens": 10,
            "total_tokens": 15,
        },
        "create_time": datetime(2024, 1, 1, tzinfo=UTC).isoformat(),
    }
