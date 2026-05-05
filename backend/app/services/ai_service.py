"""
AI service - Vietnamese text summarization using SBERT + Qwen 2.5-3B-Instruct.
Pipeline adapted from summarize_ai.ipynb:
    1. Clean markdown & chunk text uniformly
    2. TextRank with SBERT embeddings to pick top-k chunks
    3. Qwen rewrites selected chunks into a fluent summary
"""

from __future__ import annotations

import asyncio
import math
import re
from typing import Any

import networkx as nx
import torch
from sentence_transformers import SentenceTransformer, util
from transformers import AutoModelForCausalLM, AutoTokenizer

from app.config import settings

# ------------------------------------------------------------------ #
#  Output cleaning helpers
# ------------------------------------------------------------------ #
def _clean_output(text: str) -> str:
    """Remove CJK chars, English-heavy lines, and known hallucination markers."""
    text = re.sub(r"[\u4e00-\u9fff\u3000-\u303f\uff00-\uffef]", "", text)

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

    hallucination_markers = [
        "titanic", "shipper", "năm xưa", "quên lãng", "ký ức dân tộc",
        "nguyễn minh sơn", "giám đốc marketing", "công ty tnhh",
    ]
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    filtered = []
    for para in paragraphs:
        lower = para.lower()
        if any(m in lower for m in hallucination_markers):
            continue
        filtered.append(para)
    text = "\n\n".join(filtered)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _extractive_fallback(top_chunks: list[dict], target_words: int) -> str:
    """Concatenate best chunks if abstractive fails."""
    combined = " ".join([c["text"] for c in top_chunks])
    words = combined.split()
    if len(words) <= target_words:
        return combined
    truncated = " ".join(words[:target_words])
    last_punct = max(truncated.rfind("."), truncated.rfind("!"), truncated.rfind("?"))
    if last_punct > len(truncated) * 0.7:
        return truncated[:last_punct + 1]
    return truncated


# ------------------------------------------------------------------ #
#  Lazy model loader (loads once on first call)
# ------------------------------------------------------------------ #
_MODELS: dict[str, Any] = {}


def _load_models() -> dict[str, Any]:
    """Load SBERT and Qwen once, cache in module-level dict."""
    if _MODELS:
        return _MODELS

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] AI Service loading models on {device.type.upper()}...")

    # Qwen model used by the DataOps notebook pipeline.
    qwen_name = settings.AI_MODEL_NAME
    qwen_tokenizer = AutoTokenizer.from_pretrained(qwen_name, trust_remote_code=True)
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    qwen_model = AutoModelForCausalLM.from_pretrained(
        qwen_name,
        dtype=torch_dtype,
        device_map="auto" if torch.cuda.is_available() else None,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )
    if not torch.cuda.is_available():
        qwen_model = qwen_model.to(device)
    qwen_model.eval()

    _MODELS.update(
        {
            "device": device,
            "qwen_tokenizer": qwen_tokenizer,
            "qwen_model": qwen_model,
        }
    )
    print("[+] AI models loaded successfully.")
    return _MODELS


# ------------------------------------------------------------------ #
#  Text preprocessing
# ------------------------------------------------------------------ #
def _clean_markdown(text: str) -> str:
    """Strip markdown syntax back to plain text."""
    text = re.sub(r"^#+\s+", "", text, flags=re.MULTILINE)
    text = re.sub(r"\*\*([^*]+)\*\*", r"\1", text)
    text = re.sub(r"\*([^*]+)\*", r"\1", text)
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)
    text = re.sub(r"```.*?```", "", text, flags=re.DOTALL)
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    return text


def _create_uniform_chunks(text: str, max_tokens: int = 150) -> list[dict[str, Any]]:
    """Split text into evenly-sized chunks (~max_tokens words each)."""
    sentences = re.split(r"(?<=[.!?])\s+", text)
    chunks: list[dict[str, Any]] = []
    current_chunk: list[str] = []
    current_count = 0

    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
        word_count = len(sentence.split())
        if current_count + word_count > max_tokens and current_count > 0:
            chunks.append(
                {
                    "chunk_index": len(chunks),
                    "text": " ".join(current_chunk),
                    "token_count": current_count,
                }
            )
            current_chunk = [sentence]
            current_count = word_count
        else:
            current_chunk.append(sentence)
            current_count += word_count

    if current_chunk:
        chunks.append(
            {
                "chunk_index": len(chunks),
                "text": " ".join(current_chunk),
                "token_count": current_count,
            }
        )
    return chunks


# ------------------------------------------------------------------ #
#  Summarization core
# ------------------------------------------------------------------ #
def _calculate_dynamic_k(target_words: int, chunk_size: int = 150, multiplier: float = 1.0) -> int:
    """How many chunks to feed the LLM based on desired summary length."""
    required_input_words = target_words * multiplier
    return math.ceil(required_input_words / chunk_size)


def _get_top_k_chunks(
    chunks: list[dict[str, Any]], k: int, sbert_model: SentenceTransformer
) -> list[dict[str, Any]]:
    """TextRank over SBERT embeddings to extract the most salient chunks."""
    texts = [c["text"] for c in chunks]
    embeddings = sbert_model.encode(texts, show_progress_bar=False)
    similarity_matrix = util.cos_sim(embeddings, embeddings).numpy()

    nx_graph = nx.from_numpy_array(similarity_matrix)
    scores = nx.pagerank(nx_graph)

    for i, chunk in enumerate(chunks):
        chunk["score"] = scores[i]

    chunks.sort(key=lambda x: x["score"], reverse=True)
    top_k = chunks[:k]
    top_k.sort(key=lambda x: x["chunk_index"])  # restore original order
    return top_k


def _rewrite_with_qwen(
    top_chunks: list[dict[str, Any]],
    target_words: int,
    tokenizer: AutoTokenizer,
    model: AutoModelForCausalLM,
    device: torch.device,
) -> str:
    """Use Qwen to rewrite selected chunks into a cohesive summary."""
    combined_text = " ".join([c["text"] for c in top_chunks])

    system_msg = "Ban la bien tap vien tom tat tai lieu tieng Viet."
    
    # Few-Shot example: A sample summary for a different article
    few_shot_example = (
        "Ví dụ bài tóm tắt chuẩn (khoảng 500 từ) cho một bài báo về chính sách giáo dục:\n\n"
        "Chính sách giáo dục quốc gia nhằm nâng cao chất lượng dạy và học. "
        "Giai đoạn 1: Phân tích hiện trạng, bao gồm thu thập dữ liệu về tình hình giáo dục hiện tại. "
        "Bước 1: Thiết lập chính sách cơ bản. Bước 2: Dự thảo kế hoạch chi tiết. "
        "Bước 3: Đánh giá tác động. Bước 4: Tham vấn chuyên gia. "
        "Bước 5: Lập kế hoạch triển khai. Bước 6: Thực hiện chính sách. "
        "Bước 7: Kiểm soát tiến độ. Bước 8: Điều chỉnh nếu cần thiết. "
        "Giai đoạn 2: Triển khai, tập trung vào đào tạo giáo viên và cải thiện cơ sở vật chất. "
        "Giai đoạn 3: Đánh giá và điều chỉnh, đảm bảo đáp ứng nhu cầu xã hội. "
        "Bài báo nhấn mạnh tầm quan trọng của giáo dục trong phát triển bền vững.\n\n"
        "Bây giờ, hãy tóm tắt bài báo này theo cùng phong cách và cấu trúc:"
    )
    
    user_msg = (
        f"{few_shot_example}\n\n"
        "Doc doan van duoi day va viet lai thanh mot bai tom tat hoan chinh, mach lac.\n\n"
        f"Yeu cau:\n"
        f"- Do dai khoang {target_words} tu\n"
        "- KHONG duoc copy-paste nguyen van tu tai lieu goc\n"
        "- Hay doc hieu, tong hop va dien dat lai (paraphrase) bang ngon tu cua ban\n"
        "- Van phai giu nguyen y nghia, so lieu, ten giai doan, ten buoc tu tai lieu goc\n"
        "- Khong them thong tin moi, khong dua vi du ngoai le\n\n"
        f"Tai lieu goc:\n{combined_text}\n\n"
        "Bai tom tat hoan chinh (viet lai, khong copy):"
    )

    messages = [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": user_msg},
    ]

    text_input = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    model_inputs = tokenizer([text_input], return_tensors="pt").to(device)

    # Increase token budget to prevent cut-off (min 1000 tokens)
    max_out_tokens = max(int(target_words * 2.5), 1000)

    with torch.no_grad():
        generated_ids = model.generate(
            **model_inputs,
            max_new_tokens=max_out_tokens,
            min_new_tokens=int(target_words * 0.8),
            do_sample=True,
            temperature=0.5,
            top_p=0.9,
            repetition_penalty=1.3,
            no_repeat_ngram_size=3,
            pad_token_id=tokenizer.eos_token_id,
        )

    generated_ids = [
        output_ids[len(input_ids):]
        for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]
    raw_text = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    raw_text = _clean_output(raw_text)

    # Fallback: too short OR fidelity check fails
    if len(raw_text.split()) < int(target_words * 0.3) or not raw_text:
        return _extractive_fallback(top_chunks, target_words)

    # Check if output is faithful to source (threshold 0.30 allows paraphrasing)
    source_words = set(re.findall(r"[a-zA-Z_\u00c0-\u024f\u1ea0-\u1eff]+", combined_text.lower()))
    output_words = set(re.findall(r"[a-zA-Z_\u00c0-\u024f\u1ea0-\u1eff]+", raw_text.lower()))
    if output_words and len(output_words & source_words) / len(output_words) < 0.30:
        return _extractive_fallback(top_chunks, target_words)

    return raw_text


# ------------------------------------------------------------------ #
#  Public async API
# ------------------------------------------------------------------ #
async def summarize_text(text: str, target_words: int = 500) -> str:
    """
    Backward-compatible wrapper.
    The active text summarizer lives in dataops_service so text and PDF use the
    same notebook-aligned TextRank + Qwen pipeline.
    """
    from app.services.dataops_service import summarize_text as summarize_dataops_text

    return await summarize_dataops_text(text, target_words=target_words)


async def chat_with_context(context: str, prompt: str) -> str:
    """
    Chat about the document using Qwen.
    """
    models = await asyncio.to_thread(_load_models)

    system_msg = (
        "Ban la mot tro ly AI thong minh. Hay tra loi cau hoi cua nguoi dung "
        "dua tren ngu canh tai lieu duoc cung cap. Neu cau tra loi khong co trong ngu canh, "
        "hay noi ro rang rang ban khong tim thay thong tin lien quan."
    )
    user_msg = f"Ngu canh tai lieu:\n{context}\n\nCau hoi: {prompt}"

    messages = [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": user_msg},
    ]

    def _generate() -> str:
        text_input = models["qwen_tokenizer"].apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        model_inputs = models["qwen_tokenizer"]([text_input], return_tensors="pt").to(
            models["device"]
        )
        with torch.no_grad():
            generated_ids = models["qwen_model"].generate(
                **model_inputs,
                max_new_tokens=256,       # reduced for CPU speed
                do_sample=False,          # greedy = much faster on CPU
                pad_token_id=models["qwen_tokenizer"].eos_token_id,
            )
        generated_ids = [
            output_ids[len(input_ids):]
            for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]
        return models["qwen_tokenizer"].batch_decode(generated_ids, skip_special_tokens=True)[0]

    return await asyncio.to_thread(_generate)
