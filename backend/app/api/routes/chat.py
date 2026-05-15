from motor.motor_asyncio import AsyncIOMotorDatabase
from fastapi import APIRouter, Depends, HTTPException, status

from app.core.database import get_database
from app.schemas.chat import ChatRequest, ChatResponse
from app.services.chat_service import ChatService
from app.services.document_service import DocumentService

router = APIRouter(prefix="/chat", tags=["chat"])
chat_service = ChatService()


@router.post("", response_model=ChatResponse)
async def chat(payload: ChatRequest, db: AsyncIOMotorDatabase = Depends(get_database)) -> ChatResponse:
    document_service = DocumentService(db)
    document = await document_service.get_document(payload.document_id)
    if not document:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Document not found")
    return chat_service.answer_question(
        document_id=payload.document_id,
        question=payload.question,
        locale=payload.locale,
        document_text=document["extracted_text"],
    )
