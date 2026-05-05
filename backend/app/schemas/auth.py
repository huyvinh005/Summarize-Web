"""
Pydantic schemas for authentication endpoints.
"""

from pydantic import BaseModel, EmailStr


# ---------- Request bodies ----------

class RegisterRequest(BaseModel):
    username: str
    email: EmailStr
    password: str


class LoginRequest(BaseModel):
    email: EmailStr
    password: str


# ---------- Response bodies ----------

class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"


class UserResponse(BaseModel):
    id: str
    username: str
    email: str
    created_at: str
