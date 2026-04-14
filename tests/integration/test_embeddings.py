from __future__ import annotations

import os

import pytest

from inertialai_python import EmbeddingModel, EmbeddingResponse, InertialAI
from inertialai_python.types import EmbeddingMultiModalInput

pytestmark = pytest.mark.skipif(
    not os.environ.get("INERTIALAI_API_KEY"),
    reason="INERTIALAI_API_KEY not set",
)


@pytest.fixture
def client() -> InertialAI:
    return InertialAI()


class TestEmbeddingsIntegration:
    def test_create_with_text(self, client: InertialAI) -> None:
        response = client.embeddings.create(
            input=[EmbeddingMultiModalInput(text="sensor reading from accelerometer")],
            model=EmbeddingModel.INERTIAL_EMBED_ALPHA,
        )
        assert isinstance(response, EmbeddingResponse)
        assert len(response.data) == 1
        assert isinstance(response.data[0].embedding, list)

    def test_create_with_time_series(self, client: InertialAI) -> None:
        response = client.embeddings.create(
            input=[EmbeddingMultiModalInput(time_series=[[1.0, 2.0, 3.0, 4.0, 5.0]])],
            model=EmbeddingModel.INERTIAL_EMBED_ALPHA,
        )
        assert isinstance(response, EmbeddingResponse)

    def test_create_multimodal(self, client: InertialAI) -> None:
        response = client.embeddings.create(
            input=[
                EmbeddingMultiModalInput(
                    time_series=[[1.0, 2.0, 3.0]],
                    text="accelerometer data",
                )
            ],
            model=EmbeddingModel.INERTIAL_EMBED_ALPHA,
        )
        assert isinstance(response, EmbeddingResponse)
