from __future__ import annotations

import httpx

DEFAULT_BASE_URL = "https://inertialai.com"
DEFAULT_TIMEOUT = httpx.Timeout(timeout=60.0, connect=5.0)
DEFAULT_MAX_RETRIES = 2
INITIAL_RETRY_DELAY = 0.5
MAX_RETRY_DELAY = 8.0
RETRYABLE_STATUS_CODES = frozenset({429, 500, 502, 503, 504})
