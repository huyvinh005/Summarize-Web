from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    document_id: str = Field(min_length=1)
    question: str = Field(min_length=1, max_length=2000)
    locale: Literal["vi", "en"] = "vi"


class ChatResponse(BaseModel):
    answer: str
    method: str
    source: str
    created_at: datetime
