"""
AI router — summarize & chat endpoints + session history.
"""

from datetime import datetime, timezone

from bson import ObjectId
from bson.errors import InvalidId
from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from app.database import get_db
from app.models.session import chat_message, session_document, session_response
from app.schemas.ai import (
    ChatRequest,
    ChatResponse,
    SessionListItem,
    SummarizeRequest,
    SummarizeResponse,
)
from app.services.ai_service import chat_with_context
from app.services.auth_service import decode_access_token
from app.services.dataops_service import summarize_pdf_document, summarize_text

router = APIRouter(prefix="/api/ai", tags=["AI"])

# Bearer token dependency
security = HTTPBearer()


def _get_user_id(credentials: HTTPAuthorizationCredentials = Depends(security)) -> str:
    """Extract and validate user_id from the Authorization header."""
    user_id = decode_access_token(credentials.credentials)
    if user_id is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token.",
        )
    return user_id


def _parse_object_id(value: str) -> ObjectId:
    try:
        return ObjectId(value)
    except (InvalidId, TypeError):
        raise HTTPException(status_code=400, detail="Invalid session id.") from None


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


# ------------------------------------------------------------------ #
#  POST /api/ai/summarize
# ------------------------------------------------------------------ #
@router.post("/summarize", response_model=SummarizeResponse)
async def summarize(body: SummarizeRequest, user_id: str = Depends(_get_user_id)):
    """
    Receive text, run AI summarisation, and persist the session.
    Replace `summarize_text()` in ai_service.py with your own pipeline.
    """
    db = get_db()

    # Generate summary via the notebook-aligned DataOps pipeline.
    summary = await summarize_text(body.text, target_words=body.target_words)
    if summary.startswith("[Error]"):
        raise HTTPException(status_code=500, detail=summary)

    # Create a title from the first 50 chars of the text
    title = body.text[:50].strip() + ("..." if len(body.text) > 50 else "")

    # Create or update session
    if body.session_id:
        session_object_id = _parse_object_id(body.session_id)
        # Update existing session
        result = await db.sessions.update_one(
            {"_id": session_object_id, "user_id": user_id},
            {
                "$set": {
                    "original_text": body.text,
                    "summary": summary,
                    "title": title,
                    "updated_at": _utc_now(),
                }
            },
        )
        if result.matched_count == 0:
            raise HTTPException(status_code=404, detail="Session not found.")
        session_id = body.session_id
    else:
        # New session
        doc = session_document(
            user_id=user_id,
            original_text=body.text,
            summary=summary,
            title=title,
        )
        result = await db.sessions.insert_one(doc)
        session_id = str(result.inserted_id)

    return SummarizeResponse(session_id=session_id, summary=summary)


# ------------------------------------------------------------------ #
#  POST /api/ai/summarize-pdf
# ------------------------------------------------------------------ #
@router.post("/summarize-pdf", response_model=SummarizeResponse)
async def summarize_pdf_endpoint(
    file: UploadFile = File(..., description="PDF file to summarize"),
    target_words: int = Form(500, ge=50, le=2000),
    user_id: str = Depends(_get_user_id),
):
    """
    Upload a PDF, extract text (with OCR for scanned pages),
    run TextRank + Qwen summarization, and persist the session.
    """
    if file.content_type != "application/pdf":
        raise HTTPException(status_code=400, detail="Only PDF files are accepted.")

    pdf_bytes = await file.read()
    if len(pdf_bytes) == 0:
        raise HTTPException(status_code=400, detail="Empty PDF file.")

    result = await summarize_pdf_document(pdf_bytes, target_words=target_words)
    summary = result["summary"]

    if summary.startswith("[Error]"):
        raise HTTPException(status_code=500, detail=summary)

    db = get_db()
    title = file.filename or "PDF Summary"
    doc = session_document(
        user_id=user_id,
        original_text=result["source_text"] or f"[PDF uploaded: {file.filename}]",
        summary=summary,
        title=title,
    )
    result = await db.sessions.insert_one(doc)
    session_id = str(result.inserted_id)

    return SummarizeResponse(session_id=session_id, summary=summary)


# ------------------------------------------------------------------ #
#  POST /api/ai/chat
# ------------------------------------------------------------------ #
@router.post("/chat", response_model=ChatResponse)
async def chat(body: ChatRequest, user_id: str = Depends(_get_user_id)):
    """
    Chat with AI about the document in the given session.
    Replace `chat_with_context()` in ai_service.py with your own pipeline.
    """
    db = get_db()

    # Fetch session to get the original document context
    session_object_id = _parse_object_id(body.session_id)
    session = await db.sessions.find_one({"_id": session_object_id})
    if not session or session.get("user_id") != user_id:
        raise HTTPException(status_code=404, detail="Session not found.")

    context = session.get("original_text", "")

    # Generate AI reply (mock by default)
    reply = await chat_with_context(context=context, prompt=body.prompt)

    # Persist both user message and AI reply in chat_history
    await db.sessions.update_one(
        {"_id": session_object_id, "user_id": user_id},
        {
            "$set": {"updated_at": _utc_now()},
            "$push": {
                "chat_history": {
                    "$each": [
                        chat_message("user", body.prompt),
                        chat_message("assistant", reply),
                    ]
                }
            }
        },
    )

    return ChatResponse(reply=reply)


# ------------------------------------------------------------------ #
#  GET /api/ai/sessions — list user's session history
# ------------------------------------------------------------------ #
@router.get("/sessions", response_model=list[SessionListItem])
async def list_sessions(user_id: str = Depends(_get_user_id)):
    """Return all sessions for the current user (most recent first)."""
    db = get_db()
    cursor = db.sessions.find(
        {"user_id": user_id},
        {"title": 1, "created_at": 1},
    ).sort("created_at", -1)

    sessions = await cursor.to_list(length=100)
    return [
        SessionListItem(
            id=str(s["_id"]),
            title=s["title"],
            created_at=s["created_at"],
        )
        for s in sessions
    ]


# ------------------------------------------------------------------ #
#  GET /api/ai/sessions/{session_id} — get full session detail
# ------------------------------------------------------------------ #
@router.get("/sessions/{session_id}")
async def get_session(session_id: str, user_id: str = Depends(_get_user_id)):
    """Return full session data including chat history."""
    db = get_db()
    session = await db.sessions.find_one({"_id": _parse_object_id(session_id)})
    if not session or session.get("user_id") != user_id:
        raise HTTPException(status_code=404, detail="Session not found.")
    return session_response(session)
