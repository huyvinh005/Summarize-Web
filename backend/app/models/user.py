"""
MongoDB document schemas for User collection.

Example document in MongoDB:
{
    "_id": ObjectId("..."),
    "username": "john_doe",
    "email": "john@example.com",
    "hashed_password": "$2b$12$...",
    "created_at": "2026-04-28T10:00:00Z"
}
"""

from datetime import datetime, timezone


def user_document(username: str, email: str, hashed_password: str) -> dict:
    """Build a new user document ready for insertion."""
    return {
        "username": username,
        "email": email,
        "hashed_password": hashed_password,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }


def user_response(user: dict) -> dict:
    """Serialise a Mongo document to a JSON-safe dict (strip password)."""
    return {
        "id": str(user["_id"]),
        "username": user["username"],
        "email": user["email"],
        "created_at": user["created_at"],
    }
