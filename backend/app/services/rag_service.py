from __future__ import annotations

import hashlib
from functools import lru_cache
from pathlib import Path
from typing import Any

import torch
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings

try:
    from langchain_chroma import Chroma
except ImportError:
    from langchain_community.vectorstores import Chroma

from app.core.config import get_settings


class RAGService:
    def __init__(self) -> None:
        self.settings = get_settings()

    @lru_cache(maxsize=1)
    def get_embeddings(self) -> HuggingFaceEmbeddings:
        device = self._embedding_device()
        return HuggingFaceEmbeddings(
            model_name=self.settings.rag_embedding_model_name,
            model_kwargs={"device": device},
            encode_kwargs={"normalize_embeddings": True},
            show_progress=False,
        )

    @lru_cache(maxsize=1)
    def get_llm_chain(self):
        if not self.settings.rag_enable_llm_answers:
            return None

        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, pipeline
            from langchain_huggingface import HuggingFacePipeline
        except Exception:
            return None

        model_path = self._resolve_qwen_model_path()
        model_id = model_path or self.settings.rag_qwen_model_id
        if not model_id:
            return None

        try:
            tokenizer = AutoTokenizer.from_pretrained(model_id)
            model_kwargs: dict[str, Any] = {}
            use_cuda = torch.cuda.is_available() and self._llm_device() != "cpu"

            if use_cuda and self.settings.rag_enable_4bit_quantization:
                try:
                    model_kwargs["quantization_config"] = BitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_use_double_quant=True,
                        bnb_4bit_quant_type="nf4",
                        bnb_4bit_compute_dtype=torch.float16,
                    )
                    model_kwargs["device_map"] = "auto"
                except Exception:
                    model_kwargs["device_map"] = "auto"
            else:
                if use_cuda:
                    model_kwargs["torch_dtype"] = torch.float16
                    model_kwargs["device_map"] = "auto"
                else:
                    model_kwargs["device_map"] = None

            model = AutoModelForCausalLM.from_pretrained(model_id, **model_kwargs)
            generator = pipeline(
                "text-generation",
                model=model,
                tokenizer=tokenizer,
                max_new_tokens=self.settings.rag_max_new_tokens,
                temperature=self.settings.rag_temperature,
                top_p=self.settings.rag_top_p,
                repetition_penalty=self.settings.rag_repetition_penalty,
                return_full_text=False,
            )
            llm = HuggingFacePipeline(pipeline=generator)
            prompt = PromptTemplate.from_template(self._prompt_template())
            return prompt | llm | StrOutputParser()
        except Exception:
            return None

    def answer(self, *, document_id: str, document_text: str, question: str) -> tuple[str, str]:
        vectorstore = self.get_or_create_vectorstore(document_id=document_id, document_text=document_text)
        retriever = vectorstore.as_retriever(search_kwargs={"k": self.settings.rag_retriever_k})
        documents = retriever.invoke(question)
        context = self.format_docs(documents)

        if not context.strip():
            return (
                "Tài liệu hiện tại không cung cấp thông tin về vấn đề này.",
                "reference-rag-empty-context",
            )

        llm_chain = self.get_llm_chain()
        if llm_chain is None:
            return self._fallback_answer(context=context, question=question)

        answer = llm_chain.invoke({"context": context, "question": question}).strip()
        if not answer:
            return self._fallback_answer(context=context, question=question)
        return answer, "reference-rag-qwen"

    def get_or_create_vectorstore(self, *, document_id: str, document_text: str) -> Chroma:
        persist_directory = self._vectorstore_dir(document_id)
        fingerprint_path = persist_directory / ".fingerprint"
        current_fingerprint = self._fingerprint(document_text)
        embeddings = self.get_embeddings()

        if persist_directory.exists() and fingerprint_path.exists():
            saved_fingerprint = fingerprint_path.read_text(encoding="utf-8").strip()
            if saved_fingerprint == current_fingerprint:
                return Chroma(
                    persist_directory=str(persist_directory),
                    embedding_function=embeddings,
                )

        if persist_directory.exists():
            for child in persist_directory.iterdir():
                if child.is_file():
                    child.unlink()
                else:
                    import shutil

                    shutil.rmtree(child)
        else:
            persist_directory.mkdir(parents=True, exist_ok=True)

        documents = self.build_documents(document_id=document_id, document_text=document_text)
        vectorstore = Chroma.from_documents(
            documents=documents,
            embedding=embeddings,
            persist_directory=str(persist_directory),
        )
        fingerprint_path.write_text(current_fingerprint, encoding="utf-8")
        return vectorstore

    def build_documents(self, *, document_id: str, document_text: str) -> list[Document]:
        chunks = self.chunk_document(document_text)
        return [
            Document(
                page_content=chunk,
                metadata={
                    "document_id": document_id,
                    "chunk_index": index,
                    "zone": "body",
                    "breadcrumbs": f"chunk-{index + 1}",
                },
            )
            for index, chunk in enumerate(chunks)
        ]

    def chunk_document(self, document_text: str) -> list[str]:
        paragraphs = [paragraph.strip() for paragraph in document_text.split("\n\n") if paragraph.strip()]
        if not paragraphs:
            return [document_text.strip()] if document_text.strip() else []

        chunks: list[str] = []
        buffer: list[str] = []
        buffer_words = 0
        chunk_size = self.settings.rag_chunk_words
        chunk_overlap = self.settings.rag_chunk_overlap_words

        for paragraph in paragraphs:
            words = paragraph.split()
            word_count = len(words)
            if word_count >= chunk_size:
                if buffer:
                    chunks.append("\n\n".join(buffer).strip())
                    buffer = []
                    buffer_words = 0
                start = 0
                while start < word_count:
                    end = min(start + chunk_size, word_count)
                    chunk = " ".join(words[start:end]).strip()
                    if chunk:
                        chunks.append(chunk)
                    if end >= word_count:
                        break
                    start = max(end - chunk_overlap, start + 1)
                continue

            if buffer_words + word_count > chunk_size and buffer:
                chunks.append("\n\n".join(buffer).strip())
                if chunk_overlap > 0:
                    overlap_words = " ".join("\n\n".join(buffer).split()[-chunk_overlap:]).strip()
                    buffer = [overlap_words] if overlap_words else []
                    buffer_words = len(overlap_words.split()) if overlap_words else 0
                else:
                    buffer = []
                    buffer_words = 0

            buffer.append(paragraph)
            buffer_words += word_count

        if buffer:
            chunks.append("\n\n".join(buffer).strip())

        return [chunk for chunk in chunks if chunk]

    @staticmethod
    def format_docs(docs: list[Document]) -> str:
        return "\n\n---\n\n".join(doc.page_content for doc in docs)

    def _fallback_answer(self, *, context: str, question: str) -> tuple[str, str]:
        answer = (
            "Tài liệu hiện tại đã được truy xuất theo ngữ nghĩa nhưng mô hình trả lời cục bộ chưa sẵn sàng. "
            "Dưới đây là ngữ cảnh phù hợp nhất để bạn tham chiếu.\n\n"
            f"Ngữ cảnh:\n{context[:2200]}\n\n"
            f"Câu hỏi: {question}"
        )
        return answer, "reference-rag-retrieval-fallback"

    def _vectorstore_dir(self, document_id: str) -> Path:
        base_dir = Path(self.settings.rag_chroma_base_dir)
        base_dir.mkdir(parents=True, exist_ok=True)
        return base_dir / document_id

    @staticmethod
    def _fingerprint(document_text: str) -> str:
        return hashlib.sha256(document_text.encode("utf-8")).hexdigest()

    def _embedding_device(self) -> str:
        preference = self.settings.rag_device_preference
        if preference == "cpu":
            return "cpu"
        if preference == "cuda" and torch.cuda.is_available():
            return "cuda"
        return "cuda" if torch.cuda.is_available() else "cpu"

    def _llm_device(self) -> str:
        preference = self.settings.rag_llm_device_preference
        if preference == "cpu":
            return "cpu"
        if preference == "cuda" and torch.cuda.is_available():
            return "cuda"
        return "cuda" if torch.cuda.is_available() else "cpu"

    def _resolve_qwen_model_path(self) -> str | None:
        configured = self.settings.rag_qwen_model_path
        if configured and Path(configured).exists():
            return configured

        project_model = Path(__file__).resolve().parents[2] / "qwen_offline"
        if project_model.exists():
            return str(project_model)
        return None

    @staticmethod
    def _prompt_template() -> str:
        return """<|im_start|>system
Bạn là một chuyên gia phân tích và tóm tắt tài liệu. Nhiệm vụ của bạn là đọc các đoạn "Ngữ cảnh" được trích xuất từ tài liệu và thực hiện yêu cầu của người dùng.
Quy tắc tối thượng:
1. Chỉ dựa vào thông tin có trong phần Ngữ cảnh.
2. Trình bày khoa học, rõ ràng.
3. Tuyệt đối không bịa đặt thêm thông tin. Nếu ngữ cảnh không đủ, hãy nói rõ: "Tài liệu hiện tại không cung cấp thông tin về vấn đề này."<|im_end|>
<|im_start|>user
Ngữ cảnh trích xuất từ tài liệu:
{context}

Yêu cầu/Câu hỏi của tôi: {question}<|im_end|>
<|im_start|>assistant
"""
