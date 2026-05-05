"""
FastAPI application entry point.

Run with:
    uvicorn main:app --reload --port 8000
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import settings
from app.database import close_db, connect_db
from app.routers import ai, auth


# ------------------------------------------------------------------ #
#  Lifespan: connect / disconnect MongoDB
# ------------------------------------------------------------------ #
@asynccontextmanager
async def lifespan(app: FastAPI):
    await connect_db()
    yield
    await close_db()


# ------------------------------------------------------------------ #
#  Create FastAPI app
# ------------------------------------------------------------------ #
app = FastAPI(
    title="AI Text Summarizer API",
    description="Backend for AI-powered text summarisation and conversational Q&A.",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS — allow the Next.js frontend to call the API
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.frontend_origins_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register routers
app.include_router(auth.router)
app.include_router(ai.router)


@app.get("/")
async def root():
    return {"message": "AI Text Summarizer API is running."}
