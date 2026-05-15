from dataclasses import dataclass

from app.core.config import get_settings
from app.services.summary_reference_summarizer import summarize_pdf_by_textrank


@dataclass
class GeneratedSummary:
    document_id: str
    source: str
    language: str
    summary: str
    method: str


class SummaryService:
    def generate_summary(self, *, document_id: str, text: str, language: str) -> GeneratedSummary:
        settings = get_settings()
        use_llm_polish = language == "vi" and settings.summary_enable_llm_polish
        result = summarize_pdf_by_textrank(
            raw_text=text,
            target_words=settings.summary_target_words,
            use_llm_polish=use_llm_polish,
            verbose=False,
        )

        if result.used_llm:
            method = "reference-textrank-qwen"
        elif result.used_fallback and use_llm_polish:
            method = "reference-textrank-fallback"
        else:
            method = "reference-textrank-extractive"

        return GeneratedSummary(
            document_id=document_id,
            source="D:/summarize_backend/summarize_backend",
            language=language,
            summary=result.final,
            method=method,
        )
