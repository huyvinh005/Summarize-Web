from __future__ import annotations

from datetime import UTC, datetime, timedelta

from fastapi import APIRouter, Depends, HTTPException, status
from motor.motor_asyncio import AsyncIOMotorDatabase

from app.core.config import get_settings
from app.core.database import get_database
from app.core.security import get_current_user

router = APIRouter(prefix="/admin", tags=["admin"])


def _parse_admin_emails(raw: str | None) -> set[str]:
    if not raw:
        return set()
    return {email.strip().lower() for email in raw.split(",") if email.strip()}


async def require_admin_user(current_user: dict = Depends(get_current_user)) -> dict:
    settings = get_settings()
    allowlist = _parse_admin_emails(settings.admin_emails)
    email = (current_user.get("email") or "").strip().lower()
    if not allowlist or email not in allowlist:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Admin access required")
    return current_user


@router.get("/kpis")
async def get_admin_kpis(
    _admin_user: dict = Depends(require_admin_user),
    db: AsyncIOMotorDatabase = Depends(get_database),
) -> dict:
    users_col = db["users"]
    docs_col = db["documents"]
    summaries_col = db["summaries"]

    now = datetime.now(UTC)
    start_today = now.replace(hour=0, minute=0, second=0, microsecond=0)
    start_7d = start_today - timedelta(days=6)

    total_users = await users_col.count_documents({})
    verified_users = await users_col.count_documents({"is_verified": True})

    total_documents = await docs_col.count_documents({})
    total_summaries = await summaries_col.count_documents({})

    summaries_today = await summaries_col.count_documents({"created_at": {"$gte": start_today}})
    summaries_7d = await summaries_col.count_documents({"created_at": {"$gte": start_7d}})

    return {
        "total_users": total_users,
        "verified_users": verified_users,
        "total_documents": total_documents,
        "total_summaries": total_summaries,
        "summaries_today": summaries_today,
        "summaries_7d": summaries_7d,
    }


@router.get("/series")
async def get_admin_series(
    _admin_user: dict = Depends(require_admin_user),
    db: AsyncIOMotorDatabase = Depends(get_database),
    days: int = 30,
) -> dict:
    if days < 7 or days > 180:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="days must be in [7, 180]")

    now = datetime.now(UTC)
    start_day = (now.replace(hour=0, minute=0, second=0, microsecond=0) - timedelta(days=days - 1))

    users_col = db["users"]
    docs_col = db["documents"]

    users_pipeline = [
        {"$match": {"created_at": {"$gte": start_day}}},
        {"$group": {"_id": {"$dateToString": {"format": "%Y-%m-%d", "date": "$created_at"}}, "count": {"$sum": 1}}},
        {"$sort": {"_id": 1}},
    ]

    summaries_pipeline = [
        {"$match": {"created_at": {"$gte": start_day}}},
        {"$group": {"_id": {"$dateToString": {"format": "%Y-%m-%d", "date": "$created_at"}}, "count": {"$sum": 1}}},
        {"$sort": {"_id": 1}},
    ]

    summaries_col = db["summaries"]

    users_series = [{"date": row["_id"], "count": row["count"]} async for row in users_col.aggregate(users_pipeline)]
    summaries_series = [
        {"date": row["_id"], "count": row["count"]} async for row in summaries_col.aggregate(summaries_pipeline)
    ]

    return {"users": users_series, "summaries": summaries_series}


@router.get("/recent")
async def get_admin_recent(
    _admin_user: dict = Depends(require_admin_user),
    db: AsyncIOMotorDatabase = Depends(get_database),
    limit: int = 10,
) -> dict:
    if limit < 1 or limit > 50:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="limit must be in [1, 50]")

    users_col = db["users"]
    docs_col = db["documents"]
    summaries_col = db["summaries"]

    recent_users = (
        await users_col.find({}, {"email": 1, "full_name": 1, "is_verified": 1, "created_at": 1})
        .sort("created_at", -1)
        .limit(limit)
        .to_list(length=limit)
    )

    recent_documents = (
        await docs_col.find(
            {},
            {
                "title": 1,
                "source_type": 1,
                "original_filename": 1,
                "created_at": 1,
                "extraction_method": 1,
            },
        )
        .sort("created_at", -1)
        .limit(limit)
        .to_list(length=limit)
    )

    recent_summaries = (
        await summaries_col.find(
            {},
            {
                "document_id": 1,
                "method": 1,
                "source": 1,
                "rating_average": 1,
                "rating_count": 1,
                "created_at": 1,
            },
        )
        .sort("created_at", -1)
        .limit(limit)
        .to_list(length=limit)
    )

    return {
        "users": [
            {
                "id": str(user.get("_id")),
                "email": user.get("email"),
                "full_name": user.get("full_name"),
                "is_verified": user.get("is_verified"),
                "created_at": user.get("created_at"),
            }
            for user in recent_users
        ],
        "documents": [
            {
                "id": str(doc.get("_id")),
                "title": doc.get("title"),
                "source_type": doc.get("source_type"),
                "original_filename": doc.get("original_filename"),
                "extraction_method": doc.get("extraction_method"),
                "created_at": doc.get("created_at"),
            }
            for doc in recent_documents
        ],
        "summaries": [
            {
                "id": str(s.get("_id")),
                "document_id": str(s.get("document_id")) if s.get("document_id") is not None else None,
                "method": s.get("method"),
                "source": s.get("source"),
                "rating_average": s.get("rating_average"),
                "rating_count": s.get("rating_count"),
                "created_at": s.get("created_at"),
            }
            for s in recent_summaries
        ],
    }


@router.get("/me")
async def get_admin_me(_admin_user: dict = Depends(require_admin_user)) -> dict:
    return {"ok": True}
