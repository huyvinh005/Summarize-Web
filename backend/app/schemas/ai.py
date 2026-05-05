"""
Pydantic schemas for AI endpoints (summarize & chat).
"""

from pydantic import BaseModel, Field


# ---------- Summarize ----------

class SummarizeRequest(BaseModel):
    text: str
    session_id: str | None = None  # optionally link to an existing session
    target_words: int = Field(default=500, ge=50, le=2000)


class SummarizeResponse(BaseModel):
    session_id: str
    summary: str


# ---------- Chat ----------

class ChatRequest(BaseModel):
    session_id: str
    prompt: str


class ChatResponse(BaseModel):
    reply: str


# ---------- Session history ----------

class SessionListItem(BaseModel):
    id: str
    title: str
    created_at: str
