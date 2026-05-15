from fastapi import APIRouter

from app.api.routes import auth, chat, summary

api_router = APIRouter()
api_router.include_router(auth.router)
api_router.include_router(summary.router)
api_router.include_router(chat.router)
