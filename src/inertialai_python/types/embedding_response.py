from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict

from .embedding_model import EmbeddingModel


class EmbeddingData(BaseModel):
    model_config = ConfigDict(extra="ignore")

    object: str
    index: int
    embedding: list[float] | str


class EmbeddingUsage(BaseModel):
    model_config = ConfigDict(extra="ignore")

    prompt_tokens: int
    total_tokens: int


class EmbeddingResponse(BaseModel):
    model_config = ConfigDict(extra="ignore")

    id: str | None = None
    object: str
    model: EmbeddingModel
    data: list[EmbeddingData]
    usage: EmbeddingUsage
    create_time: datetime
