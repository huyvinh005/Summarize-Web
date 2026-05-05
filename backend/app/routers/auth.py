"""
Authentication router — register & login.
"""

from fastapi import APIRouter, HTTPException, status
from pymongo.errors import PyMongoError, ServerSelectionTimeoutError

from app.database import get_db
from app.models.user import user_document, user_response
from app.schemas.auth import (
    LoginRequest,
    RegisterRequest,
    TokenResponse,
    UserResponse,
)
from app.services.auth_service import create_access_token, hash_password, verify_password

router = APIRouter(prefix="/api/auth", tags=["Authentication"])


def _database_unavailable_error() -> HTTPException:
    return HTTPException(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        detail="Cannot connect to MongoDB. Check your database URL or network.",
    )


@router.post("/register", response_model=UserResponse, status_code=status.HTTP_201_CREATED)
async def register(body: RegisterRequest):
    """Register a new user and store in MongoDB."""
    db = get_db()

    # Check if email already exists
    try:
        existing = await db.users.find_one({"email": body.email})
    except ServerSelectionTimeoutError as exc:
        raise _database_unavailable_error() from exc
    if existing:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="A user with this email already exists.",
        )

    # Insert new user
    doc = user_document(
        username=body.username,
        email=body.email,
        hashed_password=hash_password(body.password),
    )
    try:
        result = await db.users.insert_one(doc)
    except PyMongoError as exc:
        raise _database_unavailable_error() from exc
    doc["_id"] = result.inserted_id

    return user_response(doc)


@router.post("/login", response_model=TokenResponse)
async def login(body: LoginRequest):
    """Authenticate user and return a JWT token."""
    db = get_db()

    try:
        user = await db.users.find_one({"email": body.email})
    except ServerSelectionTimeoutError as exc:
        raise _database_unavailable_error() from exc
    if not user or not verify_password(body.password, user["hashed_password"]):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid email or password.",
        )

    token = create_access_token(user_id=str(user["_id"]))
    return TokenResponse(access_token=token)
