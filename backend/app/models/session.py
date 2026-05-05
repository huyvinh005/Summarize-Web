"""
MongoDB document schemas for Session / Summary + Chat history.

Example Session document:
{
    "_id": ObjectId("..."),
    "user_id": "665f...",
    "title": "Article about climate change",
    "original_text": "Full article text ...",
    "summary": "AI-generated summary ...",
    "chat_history": [
        {"role": "user", "content": "What are the key points?", "timestamp": "..."},
        {"role": "assistant", "content": "The key points are ...", "timestamp": "..."}
    ],
    "created_at": "2026-04-28T10:00:00Z",
    "updated_at": "2026-04-28T10:05:00Z"
}
"""

from datetime import datetime, timezone


def session_document(
    user_id: str,
    original_text: str,
    summary: str = "",
    title: str = "Untitled Session",
) -> dict:
    """Build a new session document ready for insertion."""
    now = datetime.now(timezone.utc).isoformat()
    return {
        "user_id": user_id,
        "title": title,
        "original_text": original_text,
        "summary": summary,
        "chat_history": [],
        "created_at": now,
        "updated_at": now,
    }


def chat_message(role: str, content: str) -> dict:
    """Build a single chat message sub-document."""
    return {
        "role": role,  # "user" | "assistant"
        "content": content,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


def session_response(session: dict) -> dict:
    """Serialise a Mongo session document to JSON-safe dict."""
    return {
        "id": str(session["_id"]),
        "user_id": session["user_id"],
        "title": session["title"],
        "original_text": session["original_text"],
        "summary": session["summary"],
        "chat_history": session.get("chat_history", []),
        "created_at": session["created_at"],
        "updated_at": session["updated_at"],
    }
