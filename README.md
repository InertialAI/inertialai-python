# inertialai-python

[![PyPI](https://img.shields.io/pypi/v/inertialai-python.svg)](https://pypi.org/project/inertialai-python/)
[![Python](https://img.shields.io/pypi/pyversions/inertialai-python.svg)](https://pypi.org/project/inertialai-python/)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](./LICENSE)

The official Python client for the [InertialAI](https://www.inertialai.com) API.

InertialAI builds foundation models for time-series and real-world sensor data. This
library wraps the InertialAI REST API and gives Python developers a typed, ergonomic way
to generate multi-modal embeddings from time-series signals, text, or both.

> **Beta.** The API surface is stable enough for real use but may change before 1.0.
> Pin to a specific version in production.

## Installation

```bash
pip install inertialai-python
```

Requires Python 3.11 or newer.

## Authentication

Create an API key in the [InertialAI app](https://app.inertialai.com) and expose it as
an environment variable:

```bash
export INERTIALAI_API_KEY="iai_..."
```

The client reads `INERTIALAI_API_KEY` automatically. You can also pass it explicitly via
the `api_key` constructor argument.

## Quickstart

### Synchronous

```python
from inertialai_python import EmbeddingModel, InertialAI
from inertialai_python.types import EmbeddingMultiModalInput

client = InertialAI()

response = client.embeddings.create(
    model=EmbeddingModel.INERTIAL_EMBED_ALPHA,
    input=[
        EmbeddingMultiModalInput(
            time_series=[
                [0.01, 0.02, 0.04, 0.03, 0.02],  # channel 1 (e.g. accel x)
                [0.00, 0.01, 0.01, 0.02, 0.01],  # channel 2 (e.g. accel y)
            ],
            text="walking, 50 Hz accelerometer",
        ),
    ],
)

vector = response.data[0].embedding
print(f"Got a {len(vector)}-dim embedding using {response.model}")
```

### Asynchronous

```python
import asyncio

from inertialai_python import AsyncInertialAI, EmbeddingModel
from inertialai_python.types import EmbeddingMultiModalInput

async def main() -> None:
    async with AsyncInertialAI() as client:
        response = await client.embeddings.create(
            model=EmbeddingModel.INERTIAL_EMBED_ALPHA,
            input=[EmbeddingMultiModalInput(text="sensor reading from accelerometer")],
        )
        print(response.data[0].embedding[:8])

asyncio.run(main())
```

## Multi-modal input

Each `EmbeddingMultiModalInput` must contain at least one of:

| Field         | Type                | Description                                                                    |
| ------------- | ------------------- | ------------------------------------------------------------------------------ |
| `time_series` | `list[list[float]]` | Multi-channel signal. Outer list = channels, inner list = samples per channel. |
| `text`        | `str`               | Natural-language description to embed alongside (or instead of) the signal.    |

You can batch many inputs in a single request by passing a list. Each returned
`EmbeddingData` carries the `index` of its corresponding input.

## Configuring the client

```python
from inertialai_python import InertialAI

client = InertialAI(
    api_key="iai_...",           # overrides INERTIALAI_API_KEY
    base_url="https://inertialai.com",
    timeout=30.0,               # seconds, or an httpx.Timeout instance
    max_retries=2,              # retries for transient errors (429, 5xx, network)
)
```

Both `InertialAI` and `AsyncInertialAI` support context-manager usage to ensure the
underlying HTTP connection is closed:

```python
with InertialAI() as client:
    ...

async with AsyncInertialAI() as client:
    ...
```

## Error handling

All API errors derive from `InertialAIError`:

```
InertialAIError
├── APIError
│   └── APIStatusError
│       ├── AuthenticationError   (401)
│       ├── ValidationError       (422)
│       ├── RateLimitError        (429)
│       └── InternalServerError   (5xx)
└── APIConnectionError
    └── APITimeoutError
```

```python
from inertialai_python import (
    AuthenticationError,
    InertialAI,
    RateLimitError,
    ValidationError,
)

client = InertialAI()

try:
    client.embeddings.create(model="inertial-embed-alpha", input=[...])
except AuthenticationError:
    print("Check your INERTIALAI_API_KEY")
except RateLimitError:
    print("Slow down — you've been rate limited")
except ValidationError as e:
    print(f"Invalid request: {e}")
```

The client automatically retries transient failures (`429`, `500`, `502`, `503`, `504`,
network errors, timeouts) with exponential backoff and jitter, honoring the `Retry-After`
header when present.

## OpenAI SDK compatibility

The InertialAI embeddings endpoint is interface-compatible with OpenAI's `/v1/embeddings`
API, so the `openai` Python client will work against it if you `json.dumps` multi-modal
inputs by hand. This library exists to give you a first-class, typed experience — no
manual serialization, native `EmbeddingMultiModalInput` models, and typed error
hierarchy.

## Links

- **Website**: <https://www.inertialai.com>
- **Documentation**: <https://docs.inertialai.com>
- **Embeddings guide**: <https://docs.inertialai.com/docs/using-the-embeddings-endpoint>
- **Dashboard / API keys**: <https://app.inertialai.com>
- **Issues**: <https://github.com/InertialAI/inertialai-python/issues>
- **Changelog**: [CHANGELOG.md](./CHANGELOG.md)
- **Contributing**: [CONTRIBUTING.md](./CONTRIBUTING.md)

## License

Apache 2.0 — see [LICENSE](./LICENSE).
