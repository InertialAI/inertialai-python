from __future__ import annotations

from pydantic import BaseModel, model_validator

from .embedding_model import EmbeddingEncodingFormat, EmbeddingModel


class EmbeddingMultiModalInput(BaseModel):
    time_series: list[list[float]] | None = None
    text: str | None = None

    @model_validator(mode="after")
    def validate_at_least_one_input(self) -> EmbeddingMultiModalInput:
        if self.time_series is None and self.text is None:
            raise ValueError("At least one of 'time_series' or 'text' must be provided")
        return self


class CreateEmbeddingRequest(BaseModel):
    input: list[EmbeddingMultiModalInput]
    model: EmbeddingModel
    dimensions: int | None = None
    encoding_format: EmbeddingEncodingFormat | None = None
