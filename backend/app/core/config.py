from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

ENV_FILE = Path(__file__).resolve().parents[2] / ".env"


class Settings(BaseSettings):
    app_name: str = "Summarize AI API"
    api_prefix: str = "/api"
    frontend_origin: str = "http://localhost:3000"
    mongo_uri: str = "mongodb://localhost:27017"
    mongo_db_name: str = "summarize_ai"
    jwt_secret: str = "change-me"
    jwt_algorithm: str = "HS256"
    access_token_expire_minutes: int = 60 * 24
    verification_code_ttl_minutes: int = 10
    smtp_host: str = "smtp.gmail.com"
    smtp_port: int = 587
    smtp_username: str | None = None
    smtp_password: str | None = None
    smtp_from_email: str = "noreply@summarize-ai.local"
    smtp_from_name: str = "Summarize AI"
    summary_model_source: str = "backend/dataops-aisumary-9.ipynb"
    rag_model_source: str = "backend/RAG.ipynb"
    upload_dir: str = "backend/uploads"
    max_upload_size_mb: int = Field(default=20, ge=1, le=100)
    summary_target_words: int = Field(default=500, ge=120, le=2000)
    summary_enable_llm_polish: bool = True
    summary_llm_model_path: str | None = None
    rag_embedding_model_name: str = "bkai-foundation-models/vietnamese-bi-encoder"
    rag_chroma_base_dir: str = "backend/uploads/chroma"
    rag_retriever_k: int = Field(default=4, ge=1, le=20)
    rag_chunk_words: int = Field(default=350, ge=100, le=2000)
    rag_chunk_overlap_words: int = Field(default=60, ge=0, le=500)
    rag_enable_llm_answers: bool = False
    rag_qwen_model_id: str = "Qwen/Qwen2.5-3B-Instruct"
    rag_qwen_model_path: str | None = None
    rag_device_preference: str = "cpu"
    rag_llm_device_preference: str = "cpu"
    rag_enable_4bit_quantization: bool = False
    rag_max_new_tokens: int = Field(default=512, ge=64, le=4096)
    rag_temperature: float = Field(default=0.1, ge=0.0, le=2.0)
    rag_top_p: float = Field(default=0.9, ge=0.0, le=1.0)
    rag_repetition_penalty: float = Field(default=1.1, ge=0.5, le=3.0)
    pdf_ocr_language: str = "vie+eng"
    pdf_ocr_dpi: int = Field(default=300, ge=72, le=600)
    pdf_max_pages_to_analyze: int = Field(default=20, ge=1, le=200)
    pdf_min_avg_chars_per_page: int = Field(default=100, ge=1)
    pdf_min_text_page_ratio: float = Field(default=0.5, ge=0.0, le=1.0)
    pdf_image_area_ratio_threshold: float = Field(default=0.8, ge=0.0, le=1.0)
    pdf_max_garbage_char_ratio: float = Field(default=0.1, ge=0.0, le=1.0)
    pdf_preserve_layout: bool = True
    pdf_max_pages: int | None = Field(default=None, ge=1, le=5000)
    admin_emails: str | None = None

    model_config = SettingsConfigDict(
        env_file=str(ENV_FILE),
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )


@lru_cache
def get_settings() -> Settings:
    return Settings()
