"""
DataOps service — PDF → TextRank → Qwen summarization pipeline.
Adapted from ai_dataops.ipynb:
    1. PDF page classification (OCR vs native text)
    2. Text extraction (OCR for scanned pages, native for text pages)
    3. Chunking (≤150 tokens)
    4. TF-IDF TextRank to select top-k chunks
    5. Qwen summarization with repetition removal
"""

from __future__ import annotations

import asyncio
import io
import re
from typing import Any, TypedDict

import numpy as np
import torch
from sklearn.metrics.pairwise import cosine_similarity

# PDF processing
try:
    import fitz  # PyMuPDF
    HAS_PYMUPDF = True
except ImportError:
    HAS_PYMUPDF = False

try:
    import pytesseract
    from PIL import Image
    HAS_OCR = True
except ImportError:
    HAS_OCR = False

# Shared model loader
from app.services.ai_service import _load_models


class SummaryResult(TypedDict):
    summary: str
    source_text: str


# ------------------------------------------------------------------ #
#  B1: PDF page classification
# ------------------------------------------------------------------ #
def classify_pages(doc: fitz.Document, threshold: float = 0.05) -> dict:
    """Classify each page as 'A' (needs OCR) or 'B' (native text)."""
    classification = {}
    for i, page in enumerate(doc):
        text = page.get_text("text").strip()
        rect = page.rect
        page_area = rect.width * rect.height
        if page_area == 0:
            classification[i] = "A"
            continue
        score = len(text) / page_area
        classification[i] = "B" if score >= threshold else "A"
    return classification


# ------------------------------------------------------------------ #
#  B2: Text extraction
# ------------------------------------------------------------------ #
def _extract_text_ocr(page: fitz.Page, lang: str = "vie+eng") -> str:
    """Render page to image and run Tesseract OCR."""
    if not HAS_OCR:
        return ""
    mat = fitz.Matrix(2.0, 2.0)
    pix = page.get_pixmap(matrix=mat)
    img_bytes = pix.tobytes("png")
    img = Image.open(io.BytesIO(img_bytes))
    text = pytesseract.image_to_string(img, lang=lang)
    return text.strip()


def _extract_text_native(page: fitz.Page) -> str:
    """Extract selectable text directly from PDF."""
    return page.get_text("text").strip()


def extract_all_text(doc: fitz.Document, classification: dict, ocr_lang: str = "vie+eng") -> list:
    """Extract text from all pages based on classification."""
    results = []
    for i, page in enumerate(doc):
        group = classification.get(i, "B")
        if group == "A" and HAS_OCR:
            text = _extract_text_ocr(page, ocr_lang)
        else:
            text = _extract_text_native(page)
        results.append({"page": i + 1, "group": group, "text": text})
    return results


# ------------------------------------------------------------------ #
#  B3: Chunking
# ------------------------------------------------------------------ #
def _simple_token_count(text: str) -> int:
    return len(text.split())


def _split_sentences(text: str) -> list[str]:
    sentences = re.split(r"(?<=[.!?।])\s+|(?<=\n)\s*", text)
    return [s.strip() for s in sentences if s.strip()]


def chunk_text(pages_data: list[dict], max_tokens: int = 150) -> list[str]:
    """Merge all pages → split sentences → create chunks ≤ max_tokens."""
    full_text = "\n".join(p["text"] for p in pages_data if p["text"])
    sentences = _split_sentences(full_text)
    chunks: list[str] = []
    current_chunk: list[str] = []
    current_count = 0

    for sent in sentences:
        sent_tokens = _simple_token_count(sent)

        # If single sentence exceeds max_tokens → split by words
        if sent_tokens > max_tokens:
            words = sent.split()
            sub_chunk = []
            sub_count = 0
            for word in words:
                if sub_count + 1 > max_tokens:
                    chunks.append(" ".join(sub_chunk))
                    sub_chunk = [word]
                    sub_count = 1
                else:
                    sub_chunk.append(word)
                    sub_count += 1
            if sub_chunk:
                remaining = " ".join(sub_chunk)
                rem_count = _simple_token_count(remaining)
                if current_count + rem_count <= max_tokens:
                    current_chunk.append(remaining)
                    current_count += rem_count
                else:
                    if current_chunk:
                        chunks.append(" ".join(current_chunk))
                    current_chunk = [remaining]
                    current_count = rem_count
            continue

        if current_count + sent_tokens <= max_tokens:
            current_chunk.append(sent)
            current_count += sent_tokens
        else:
            if current_chunk:
                chunks.append(" ".join(current_chunk))
            current_chunk = [sent]
            current_count = sent_tokens

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks


# ------------------------------------------------------------------ #
#  B4: TF-IDF TextRank
# ------------------------------------------------------------------ #
def _build_tfidf_matrix(chunks: list[str]) -> np.ndarray:
    tokenized = [chunk.lower().split() for chunk in chunks]
    vocab = list(set(w for tokens in tokenized for w in tokens))
    if not vocab:
        return np.zeros((len(chunks), 1))
    word2idx = {w: i for i, w in enumerate(vocab)}

    tf_matrix = np.zeros((len(chunks), len(vocab)))
    for i, tokens in enumerate(tokenized):
        for word in tokens:
            tf_matrix[i][word2idx[word]] += 1
        if len(tokens) > 0:
            tf_matrix[i] /= len(tokens)

    df = np.sum(tf_matrix > 0, axis=0)
    idf = np.log((len(chunks) + 1) / (df + 1)) + 1
    return tf_matrix * idf


def textrank(chunks: list[str], damping: float = 0.85, max_iter: int = 100, tol: float = 1e-4) -> np.ndarray:
    """TextRank over TF-IDF vectors."""
    if len(chunks) == 0:
        return np.array([])

    tfidf = _build_tfidf_matrix(chunks)
    sim_matrix = cosine_similarity(tfidf)

    row_sums = sim_matrix.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    sim_matrix = sim_matrix / row_sums

    n = len(chunks)
    scores = np.ones(n) / n

    for _ in range(max_iter):
        new_scores = (1 - damping) / n + damping * sim_matrix.T @ scores
        if np.abs(new_scores - scores).sum() < tol:
            break
        scores = new_scores

    return scores


def compute_k(target_word_count: int, max_tokens_per_chunk: int = 150, input_limit: int = 8192) -> int:
    """How many chunks to feed the LLM based on desired output length."""
    tokens_needed = int(target_word_count * 5)
    k = (tokens_needed + max_tokens_per_chunk - 1) // max_tokens_per_chunk
    max_k = input_limit // max_tokens_per_chunk
    k = min(k, max_k)
    return max(k, 2)


def select_top_k_chunks(chunks: list[str], scores: np.ndarray, k: int) -> list[str]:
    """Select top-k chunks by TextRank score, preserving document order.

    Keep the opening chunks because academic PDFs often state the objective,
    scope, and structure there; pure TextRank can drop that context.
    """
    if len(chunks) <= k:
        return chunks
    opening_indices = set(range(min(2, len(chunks), k)))
    remaining_k = max(k - len(opening_indices), 0)
    ranked_indices = [int(i) for i in np.argsort(scores)[::-1]]
    top_indices = list(opening_indices)
    for idx in ranked_indices:
        if idx not in opening_indices:
            top_indices.append(idx)
        if len(top_indices) >= len(opening_indices) + remaining_k:
            break
    top_indices_sorted = sorted(top_indices)
    return [chunks[i] for i in top_indices_sorted]


# ------------------------------------------------------------------ #
#  B5+B6: Summarization with shared Qwen model
# ------------------------------------------------------------------ #
def _normalize_for_repetition(text: str) -> str:
    return re.sub(r"\W+", " ", text.lower()).strip()


def _remove_repetition(text: str) -> str:
    """Remove repeated paragraphs/sentences and restart markers."""
    restart_parts = re.split(r"(?i)\btóm\s*tắt\s*hoàn\s*chỉnh\s*:?", text, maxsplit=1)
    if len(restart_parts) > 1:
        prefix = restart_parts[0].strip()
        suffix = restart_parts[1].strip()
        text = prefix if len(prefix.split()) >= 30 else suffix

    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    if len(paragraphs) > 1:
        seen, result = set(), []
        for para in paragraphs:
            fingerprint = _normalize_for_repetition(para)[:120]
            if fingerprint in seen:
                break
            seen.add(fingerprint)
            result.append(para)
        text = "\n\n".join(result)

    sentences = re.split(r"(?<=[.!?])\s+", text)
    seen_sentences, kept = set(), []
    for sentence in sentences:
        normalized = sentence.strip()
        if not normalized:
            continue
        fingerprint = _normalize_for_repetition(normalized)[:140]
        if fingerprint in seen_sentences:
            continue
        seen_sentences.add(fingerprint)
        kept.append(normalized)
    return " ".join(kept).strip()


_META_COMMENTARY_PATTERNS = [
    r"\b(?:bài viết|bản tóm tắt|đoạn tóm tắt)\s+(?:này\s+)?(?:sẽ|tập trung|trình bày|tóm tắt)\b",
    r"\b(?:dưới đây|sau đây)\s+là\b",
    r"\bvì\s+yêu\s+cầu\s+(?:là\s+)?(?:viết\s+)?(?:khoảng\s+)?\d+\s+từ\b",
    r"\btuyệt\s+đối\s+không\b",
    r"\bcác\s+quy\s+tắc\s+cấm\s+kỵ\b",
    r"\bquy\s+tắc\s+nghiêm\s+ngặt\b",
    r"\bkhông\s+đưa\s+bất\s+kỳ\s+quy\s+tắc\b",
    r"\bkhông\s+in\s+lại\s+tên\s+thẻ\b",
    r"\bchỉ\s+in\s+ra\s+nội\s+dung\s+bài\s+tóm\s+tắt\b",
    r"\bbắt\s+tay\s+ngay\s+vào\b",
    r"\bkhông\s+bao\s+gồm\s+các\s+chi\s+tiết\b",
    r"\btài\s+liệu\s+tham\s+khảo\b",
    r"\bngày\s+nhận\s+bài\b",
    r"\bngày\s+chấp\s+nhận\b",
]


def _remove_meta_commentary(text: str) -> str:
    """Remove assistant self-commentary and reference-list leakage."""
    text = re.split(r"(?i)\btài\s+liệu\s+tham\s+khảo\s*:?", text, maxsplit=1)[0]
    text = re.sub(r"(?i)\btóm\s*tắt\s*hoàn\s*chỉnh\s*:?", " ", text)
    text = re.sub(r"<[^>]+>", " ", text)
    sentences = re.split(r"(?<=[.!?])\s+", text)
    kept = []
    for sentence in sentences:
        normalized = sentence.strip()
        if not normalized:
            continue
        lower = normalized.lower()
        if any(re.search(pattern, lower, flags=re.I) for pattern in _META_COMMENTARY_PATTERNS):
            continue
        kept.append(normalized)
    return " ".join(kept).strip()


def _clean_output(text: str, source_text: str = "") -> str:
    """
    Post-process model output:
    - Remove CJK characters (Chinese, Japanese, Korean)
    - Remove lines with excessive English words
    - Remove known hallucination markers
    - Truncate if output drifts from source topic
    """
    # 1. Remove CJK characters
    text = re.sub(r"[\u4e00-\u9fff\u3000-\u303f\uff00-\uffef]", "", text)

    # 2. Remove lines that are mostly English (more than 50% Latin words)
    lines = text.split("\n")
    cleaned_lines = []
    for line in lines:
        words = line.split()
        if not words:
            continue
        latin_words = [w for w in words if re.match(r"^[a-zA-Z]+$", w)]
        if len(latin_words) / len(words) > 0.5:
            continue
        cleaned_lines.append(line)
    text = "\n".join(cleaned_lines)

    # 3. Remove known hallucination markers / off-topic paragraphs
    text = re.sub(
        r"^(văn bản hoàn chỉnh|văn bản tóm tắt hoàn chỉnh|bài tóm tắt hoàn chỉnh)\s*:?",
        "",
        text,
        flags=re.I,
    )

    # 4. Clean up extra whitespace
    text = _remove_meta_commentary(text)
    text = _remove_repetition(text)
    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r"(?:^|\s)(?:\d+|[A-Za-z])\.\s*$", "", text).strip()

    return text


def _extractive_fallback(top_k_chunks: list[str], target_word_count: int) -> str:
    """Fallback: concatenate best chunks into coherent paragraphs."""
    combined = " ".join(top_k_chunks)
    words = combined.split()
    if len(words) <= target_word_count:
        return combined
    # Take first target_word_count words, end at sentence boundary
    truncated = " ".join(words[:target_word_count])
    last_punct = max(truncated.rfind("."), truncated.rfind("!"), truncated.rfind("?"))
    if last_punct > len(truncated) * 0.7:
        return truncated[:last_punct + 1]
    return truncated


_VIETNAMESE_STOPWORDS = {
    "va", "cua", "la", "mot", "cac", "trong", "duoc", "voi", "cho", "de", "ve", "khong",
    "co", "du", "tu", "nhung", "thi", "ma", "con", "da", "hoac", "the", "nay", "do",
    "o", "vi", "neu", "nhu", "khi", "sau", "truoc", "tai", "den", "qua", "roi", "cung",
    "rat", "rat", "nhieu", "it", "hon", "bang", "theo", "vao", "ra", "nhu", "tren", "duoi",
    "giua", "ben", "noi", "chua", "phai", "nen", "vi", "boi", "tuy", "mac", "du",
}


def _output_fidelity_check(output: str, source: str, threshold: float = 0.55) -> bool:
    """
    Check if the output is faithful to the source.
    Returns True if output passes the fidelity check.
    Measures % of output content words that appear in source.
    """
    if not output or not source:
        return False

    def _extract_words(text: str) -> set[str]:
        words = re.findall(r"[a-zA-Z_\u00c0-\u024f\u1ea0-\u1eff]+", text.lower())
        return {w for w in words if w not in _VIETNAMESE_STOPWORDS and len(w) > 2}

    source_words = _extract_words(source)
    output_words = _extract_words(output)

    if not output_words:
        return False

    overlap = len(output_words & source_words) / len(output_words)
    return overlap >= threshold


def summarize_with_dataops(
    top_k_chunks: list[str],
    target_word_count: int,
    tokenizer: Any,
    model: Any,
    device: Any,
) -> str:
    """DataOps-style summarization using shared Qwen model."""
    context = "\n\n".join(
        [f"[Đoạn {i+1}]: {chunk}" for i, chunk in enumerate(top_k_chunks)]
    )
    prompt = (
        "Bạn là chuyên gia tóm tắt tài liệu tiếng Việt. "
        f"Hãy viết một bản tóm tắt khoảng {target_word_count} từ từ tài liệu nguồn.\n\n"
        "Yêu cầu bắt buộc:\n"
        "- Chỉ xuất ra nội dung tóm tắt cuối cùng, không tự bình luận hoặc nhắc lại yêu cầu.\n"
        "- Giữ đúng thông tin cốt lõi, số liệu, tên riêng, quan hệ nguyên nhân-kết quả từ tài liệu nguồn.\n"
        "- Không thêm thông tin ngoài tài liệu nguồn, không suy diễn.\n"
        "- Diễn đạt mạch lạc bằng tiếng Việt tự nhiên, tránh lặp ý và tránh chép nguyên văn dài.\n"
        "- Tổ chức nội dung theo mạch: mở đầu nêu chủ đề, thân bài tổng hợp ý chính theo logic tài liệu, kết luận nêu thông điệp trung tâm.\n\n"
        f"Tài liệu nguồn:\n{context}\n\n"
        "Bản tóm tắt:"
    )

    messages = [
        {
            "role": "system",
            "content": (
                "Bạn là chuyên gia tóm tắt tài liệu. Chỉ xuất ra bài tóm tắt cuối cùng. "
                "Không xuất thẻ XML, không xuất quy tắc, không trò chuyện, không xin lỗi, "
                "không tự bình luận về yêu cầu và không đưa tài liệu tham khảo."
            ),
        },
        {"role": "user", "content": prompt},
    ]

    text_input = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    model_inputs = tokenizer([text_input], return_tensors="pt").to(device)

    # Give the model enough room to finish instead of ending at dangling markers like "3.".
    dynamic_max_tokens = max(int(target_word_count * 3), 1000)
    min_new_tokens = max(int(target_word_count * 1.15), 300)

    with torch.no_grad():
        output_ids = model.generate(
            **model_inputs,
            max_new_tokens=dynamic_max_tokens,
            min_new_tokens=min(min_new_tokens, dynamic_max_tokens - 1),
            do_sample=False,
            repetition_penalty=1.15,
            no_repeat_ngram_size=5,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id,
        )

    generated = output_ids[0][model_inputs["input_ids"].shape[1]:]
    raw_text = tokenizer.decode(generated, skip_special_tokens=True).strip()

    # Post-processing
    raw_text = _remove_repetition(raw_text)
    raw_text = _clean_output(raw_text, context)

    words = raw_text.split()
    actual = len(words)

    if not raw_text:
        return "[Error] Model did not generate a summary. Try a smaller target word count or a larger model."

    if len(words) < max(int(target_word_count * 0.5), 120) or not _output_fidelity_check(raw_text, context, threshold=0.45):
        return _extractive_fallback(top_k_chunks, target_word_count)

    if actual > int(target_word_count * 1.1):
        truncated = " ".join(words[:target_word_count])
        last_punct = max(truncated.rfind("."), truncated.rfind("!"), truncated.rfind("?"))
        if last_punct > len(truncated) * 0.7:
            raw_text = truncated[:last_punct + 1]
        else:
            raw_text = truncated

    return raw_text.strip()


def _clean_plain_text(text: str) -> str:
    """Normalize pasted text before chunking."""
    text = re.sub(r"```.*?```", " ", text, flags=re.DOTALL)
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)
    text = re.sub(r"[*_`>#-]+", " ", text)
    text = re.sub(r"\r\n?", "\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def summarize_extracted_text(
    text: str,
    target_words: int,
    tokenizer: Any,
    model: Any,
    device: Any,
) -> str:
    """Run the notebook-aligned DataOps pipeline on already extracted text."""
    clean_text = _clean_plain_text(text)
    pages_data = [{"page": 1, "group": "B", "text": clean_text}]
    chunks = chunk_text(pages_data, max_tokens=150)
    if not chunks:
        return "[Error] Input text is too short or empty after cleaning."

    scores = textrank(chunks)
    k = compute_k(target_words, max_tokens_per_chunk=150, input_limit=8192)
    top_k_chunks = select_top_k_chunks(chunks, scores, k)
    return summarize_with_dataops(top_k_chunks, target_words, tokenizer, model, device)


async def summarize_text(text: str, target_words: int = 500) -> str:
    """Public API for plain text summarization using the same DataOps pipeline as PDF."""
    models = await asyncio.to_thread(_load_models)
    return await asyncio.to_thread(
        summarize_extracted_text,
        text,
        target_words,
        models["qwen_tokenizer"],
        models["qwen_model"],
        models["device"],
    )


# ------------------------------------------------------------------ #
#  Public async API
# ------------------------------------------------------------------ #
async def summarize_pdf(
    pdf_bytes: bytes,
    target_words: int = 500,
    ocr_threshold: float = 0.05,
    ocr_lang: str = "vie+eng",
) -> str:
    result = await summarize_pdf_document(
        pdf_bytes,
        target_words=target_words,
        ocr_threshold=ocr_threshold,
        ocr_lang=ocr_lang,
    )
    return result["summary"]


async def summarize_pdf_document(
    pdf_bytes: bytes,
    target_words: int = 500,
    ocr_threshold: float = 0.05,
    ocr_lang: str = "vie+eng",
) -> SummaryResult:
    """
    Full DataOps pipeline for PDF summarization.
    Runs sync code in thread pool to keep FastAPI responsive.
    """
    if not HAS_PYMUPDF:
        return {
            "summary": "[Error] PyMuPDF is not installed. Run: pip install pymupdf",
            "source_text": "",
        }

    models = await asyncio.to_thread(_load_models)
    device = models["device"]
    tokenizer = models["qwen_tokenizer"]
    model = models["qwen_model"]

    def _process() -> SummaryResult:
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        try:
            # B1: Classify pages
            classification = classify_pages(doc, ocr_threshold)

            # B2: Extract text
            pages_data = extract_all_text(doc, classification, ocr_lang)
            source_text = "\n\n".join(p["text"] for p in pages_data if p["text"]).strip()

            # B3: Chunking
            chunks = chunk_text(pages_data, max_tokens=150)
            if not chunks:
                return {
                    "summary": "[Error] No text extracted from PDF.",
                    "source_text": source_text,
                }

            # B4: TextRank + select top-k
            scores = textrank(chunks)
            k = compute_k(target_words, max_tokens_per_chunk=150, input_limit=8192)
            top_k_chunks = select_top_k_chunks(chunks, scores, k)

            # B5+B6: Summarize
            summary = summarize_with_dataops(
                top_k_chunks, target_words, tokenizer, model, device
            )
            return {"summary": summary, "source_text": source_text}
        finally:
            doc.close()

    return await asyncio.to_thread(_process)
