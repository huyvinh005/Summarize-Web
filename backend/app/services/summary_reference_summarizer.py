from __future__ import annotations

import math
import os
import re
import time
import unicodedata
from collections import Counter
from dataclasses import dataclass
from typing import Optional

import numpy as np

os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")

try:
    import torch
except Exception:
    torch = None

try:
    from unsloth import FastLanguageModel
except Exception:
    FastLanguageModel = None

SUM_MODEL_NAME = "./qwen_offline"
SUM_MAX_SEQ_LENGTH = 8192
sum_model = None
sum_tokenizer = None
_model_load_attempted = False

VI_STOPWORDS = {
    "và", "là", "của", "có", "cho", "trong", "được", "với", "các", "những",
    "một", "này", "đó", "khi", "từ", "đến", "trên", "dưới", "về", "theo",
    "tại", "bởi", "do", "nên", "đã", "đang", "sẽ", "rằng", "thì", "mà",
    "như", "hoặc", "nếu", "để", "ra", "vào", "sau", "trước", "giữa",
    "bị", "bằng", "không", "cũng", "nhiều", "ít", "hơn", "nhất", "gồm",
    "qua", "lại", "năm", "ngày", "tháng", "bài", "báo", "nghiên", "cứu",
    "tỷ", "lệ", "kết", "quả",
}

IMPORTANT_TERMS = {
    "mục tiêu", "phương pháp", "kết quả", "kết luận", "tóm tắt", "tổng quan",
    "tóm lại", "nhìn chung", "giai đoạn", "chính sách", "tổ chức",
    "quản lý", "biến động", "lãnh thổ", "phát triển", "thực thi", "chủ quyền",
    "ảnh hưởng", "vai trò", "ý nghĩa", "đánh giá", "tác động", "đặc điểm",
    "đối tượng", "quá trình", "phân tích",
}

SECTION_HEADINGS = [
    "ĐẶT VẤN ĐỀ", "MỞ ĐẦU", "GIỚI THIỆU", "ĐỐI TƯỢNG VÀ PHƯƠNG PHÁP",
    "PHƯƠNG PHÁP NGHIÊN CỨU", "KẾT QUẢ", "BÀN LUẬN", "KẾT LUẬN", "TỔNG KẾT",
]


def _resolve_model_dir() -> str:
    current_dir = os.path.dirname(__file__)
    candidate = os.path.abspath(os.path.join(current_dir, "..", "..", "qwen_offline"))
    if os.path.isdir(candidate):
        return candidate
    return SUM_MODEL_NAME


def ensure_sum_model() -> bool:
    global sum_model, sum_tokenizer, _model_load_attempted
    if sum_model is not None and sum_tokenizer is not None:
        return True
    if _model_load_attempted:
        return False
    _model_load_attempted = True
    if FastLanguageModel is None or torch is None:
        return False
    model_dir = _resolve_model_dir()
    if not os.path.isdir(model_dir):
        return False
    try:
        loaded_model, loaded_tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_dir,
            max_seq_length=SUM_MAX_SEQ_LENGTH,
            dtype=None,
            load_in_4bit=True,
        )
        FastLanguageModel.for_inference(loaded_model)
        if loaded_tokenizer.pad_token_id is None:
            loaded_tokenizer.pad_token = loaded_tokenizer.eos_token
        sum_model = loaded_model
        sum_tokenizer = loaded_tokenizer
        return True
    except Exception:
        sum_model = None
        sum_tokenizer = None
        return False


def _nfc(text: str) -> str:
    return unicodedata.normalize("NFC", text or "")


def _strip_accents(text: str) -> str:
    text = unicodedata.normalize("NFD", text)
    text = "".join(ch for ch in text if unicodedata.category(ch) != "Mn")
    return text.replace("đ", "d").replace("Đ", "D")


def _squeeze(text: str) -> str:
    text = _nfc(text or "")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\s+([,.;:!?%])", r"\1", text)
    text = re.sub(r"([(])\s+", r"\1", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def word_count(text: str) -> int:
    return len((text or "").split())


def extract_title(raw_text: str) -> str:
    text = _nfc(raw_text)
    lines = []
    for line in text.splitlines():
        line = _squeeze(line)
        if len(line.split()) >= 8 and not re.search(r"doi|tạp chí|bản quyền", line, re.I):
            lines.append(line)
    return lines[0] if lines else "Tài liệu khoa học/Nghiên cứu"


def remove_abstract_blocks(text: str) -> str:
    text = re.sub(r"(?is)\bTÓM TẮT\b.*?\bTừ kh[oó]a\b.*?(?=\bĐẶT VẤN ĐỀ\b|\bI\.\s*ĐẶT VẤN ĐỀ\b|\b1\.\s*ĐẶT VẤN ĐỀ\b|\bGIỚI THIỆU\b)", " ", text)
    text = re.sub(r"(?is)\bABSTRACT\s*:?.*?\bKeywords?\b.*?(?=\bTÓM TẮT\b|\bĐẶT VẤN ĐỀ\b|\bI\.\s*ĐẶT VẤN ĐỀ\b|\bGIỚI THIỆU\b)", " ", text)
    return text


def clean_pdf_body_text(raw_text: str) -> str:
    text = _nfc(raw_text).replace("\r", "\n")
    text = remove_abstract_blocks(text)
    text = re.split(r"(?im)^\s*(TÀI LIỆU THAM KHẢO|REFERENCES)\s*$", text)[0]

    body_start = None
    for pattern in [r"(?im)^\s*(I\.|1\.)?\s*ĐẶT VẤN ĐỀ\s*$", r"(?im)^\s*(I\.|1\.)?\s*MỞ ĐẦU\s*$", r"(?im)^\s*(I\.|1\.)?\s*GIỚI THIỆU\s*$"]:
        match = re.search(pattern, text)
        if match:
            body_start = match.start()
            break
    if body_start is not None:
        text = text[body_start:]

    noise_patterns = [r"^Tạp chí Khoa học", r"^Tập\s+\d+", r"^Bản quyền", r"^DOI\s*:", r"^https?://", r"^\*?Tác giả liên hệ", r"^Điện thoại\s*:", r"^Email\s*:", r"^Thông tin bài đăng", r"^Ngày nhận bài\s*:", r"^Ngày phản biện\s*:", r"^Ngày duyệt bài\s*:", r"^PGS\.?TS", r"^TS\.", r"^ThS\.", r"^GS\.", r"^BS\.", r"^Viện\s+", r"^Trường Đại học"]
    kept = []
    for line in text.splitlines():
        line = _squeeze(line)
        if not line:
            kept.append("")
            continue
        if re.fullmatch(r"\d{1,3}", line):
            continue
        if any(re.search(pattern, line, flags=re.IGNORECASE) for pattern in noise_patterns):
            continue
        letters = len(re.findall(r"[A-Za-zÀ-ỹĐđ]", line))
        digits = len(re.findall(r"\d", line))
        if len(line.split()) <= 4 and digits > letters:
            continue
        kept.append(line)

    text = "\n".join(kept)
    text = re.sub(r"(?<=\w)-\n(?=\w)", "", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"(?<!\n)\n(?!\n)", " ", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    return _squeeze(text)


def protect_abbreviations(text: str) -> str:
    protected = {"cs.": "cs<dot>", "ThS.": "ThS<dot>", "TS.": "TS<dot>", "BS.": "BS<dot>", "PGS.": "PGS<dot>", "GS.": "GS<dot>", "vs.": "vs<dot>", "v.v.": "vv<dot>"}
    for key, value in protected.items():
        text = text.replace(key, value)
    return text


def unprotect_abbreviations(text: str) -> str:
    return text.replace("<dot>", ".")


def split_sentences(text: str) -> list[str]:
    text = protect_abbreviations(_squeeze(text))
    for heading in SECTION_HEADINGS:
        text = re.sub(rf"\b{re.escape(heading)}\b", f". {heading}. ", text, flags=re.IGNORECASE)

    raw_sentences = re.split(r"(?<=[.!?])\s+(?=[A-ZÀ-ỸĐ0-9])", text)
    sentences = []
    for sentence in raw_sentences:
        sentence = unprotect_abbreviations(_squeeze(sentence))
        if not sentence:
            continue
        sentence_words = word_count(sentence)
        if sentence_words < 7 or sentence_words > 90:
            continue
        if re.search(r"(?i)\b(tạp chí|doi|bản quyền|email|điện thoại)\b", sentence):
            continue
        if re.search(r"\bKIẾN NGHỊ\b", sentence, flags=re.IGNORECASE):
            sentence = re.split(r"\bKIẾN NGHỊ\b", sentence, flags=re.IGNORECASE)[0].strip()
            if word_count(sentence) < 7:
                continue
        table_markers = [r"nguyên nhân\s*\( ?% ?\)", r"\bBảng\s+\d+", r"\bBiểu đồ\s+\d+", r"\bHình\s+\d+", r"\(N\s*="]
        if sum(1 for pattern in table_markers if re.search(pattern, sentence, flags=re.IGNORECASE)) >= 2:
            continue
        alpha = len(re.findall(r"[A-Za-zÀ-ỹĐđ]", sentence))
        if alpha < 20:
            continue
        sentences.append(sentence)

    output = []
    seen = set()
    for sentence in sentences:
        key = _strip_accents(sentence.lower())
        key = re.sub(r"[^a-z0-9à-ỹđ]+", " ", key)
        key = " ".join(key.split()[:18])
        if key in seen:
            continue
        seen.add(key)
        output.append(sentence)
    return output


def tokenize_for_rank(sentence: str) -> list[str]:
    normalized = _strip_accents(sentence.lower())
    words = re.findall(r"[a-zA-ZÀ-ỹĐđ]{2,}", normalized)
    return [word for word in words if word not in VI_STOPWORDS and len(word) >= 2]


def build_tfidf_matrix(sentences: list[str], max_terms: int = 1200) -> np.ndarray:
    tokenized = [tokenize_for_rank(sentence) for sentence in sentences]
    document_frequency = Counter()
    term_frequencies = []
    for tokens in tokenized:
        frequency = Counter(tokens)
        term_frequencies.append(frequency)
        document_frequency.update(frequency.keys())

    terms = [term for term, count in document_frequency.most_common(max_terms) if count >= 2 or len(sentences) < 30]
    vocabulary = {term: index for index, term in enumerate(terms)}
    total_sentences = len(sentences)
    matrix = np.zeros((total_sentences, len(vocabulary)), dtype=np.float32)

    for row_index, frequency in enumerate(term_frequencies):
        for term, count in frequency.items():
            column_index = vocabulary.get(term)
            if column_index is None:
                continue
            idf = math.log((1 + total_sentences) / (1 + document_frequency[term])) + 1.0
            matrix[row_index, column_index] = (1.0 + math.log(count)) * idf

    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return matrix / norms


def pagerank_scores(similarity: np.ndarray, damping: float = 0.85, max_iter: int = 100, tol: float = 1e-6) -> np.ndarray:
    total = similarity.shape[0]
    if total == 0:
        return np.array([])
    weights = similarity.copy()
    np.fill_diagonal(weights, 0.0)
    weights[weights < 0.05] = 0.0
    row_sums = weights.sum(axis=1, keepdims=True)
    weights = np.divide(weights, row_sums, out=np.zeros_like(weights), where=row_sums != 0)
    scores = np.ones(total, dtype=np.float32) / total
    base = (1.0 - damping) / total
    for _ in range(max_iter):
        new_scores = base + damping * (weights.T @ scores)
        if np.linalg.norm(new_scores - scores, ord=1) < tol:
            scores = new_scores
            break
        scores = new_scores
    return scores


def section_bonus(sentence: str) -> float:
    sentence_lower = sentence.lower()
    bonus = 1.0
    if any(term in sentence_lower for term in IMPORTANT_TERMS):
        bonus += 0.18
    if re.search(r"(?i)\b(mục tiêu|phương pháp|kết quả|kết luận|tóm tắt|tổng quan)\b", sentence_lower):
        bonus += 0.08
    return bonus


def clean_sentence_for_display(sentence: str) -> str:
    sentence = _squeeze(sentence)
    sentence = re.split(r"\bKIẾN NGHỊ\b", sentence, flags=re.IGNORECASE)[0]
    sentence = re.sub(r"\s*\(\d+\)\s*", " ", sentence)
    sentence = re.sub(r"\s*\[\d+(?:\s*,\s*\d+)*\]\s*", " ", sentence)
    sentence = re.sub(r"\b(BÀN LUẬN|KẾT LUẬN|KẾT QUẢ|PHƯƠNG PHÁP NGHIÊN CỨU)\b", " ", sentence, flags=re.IGNORECASE)
    sentence = re.sub(r"\s+([,.;:!?%)])", r"\1", sentence)
    sentence = re.sub(r"\.{2,}", ".", sentence)
    sentence = re.sub(r"\s{2,}", " ", sentence)
    return _squeeze(sentence)


def select_sentences_textrank(sentences: list[str], base_scores: np.ndarray, target_words: int = 500, min_words: int = 320, redundancy_threshold: float = 0.72) -> list[dict]:
    if not sentences:
        return []
    matrix = build_tfidf_matrix(sentences)
    similarity = matrix @ matrix.T
    final_scores = np.array([float(base_scores[index]) * section_bonus(sentences[index]) for index in range(len(sentences))], dtype=np.float32)
    order = list(np.argsort(-final_scores))
    selected = []
    selected_ids = []
    total_words = 0

    def is_redundant(index: int, threshold: float = redundancy_threshold) -> bool:
        if not selected_ids:
            return False
        return max(float(similarity[index, selected_index]) for selected_index in selected_ids) > threshold

    def add_index(index: int, *, allow_over: bool = False, threshold: float = redundancy_threshold) -> bool:
        nonlocal total_words
        if index in selected_ids:
            return False
        raw_sentence = sentences[index]
        cleaned_sentence = clean_sentence_for_display(raw_sentence)
        if not cleaned_sentence:
            return False
        sentence_words = word_count(cleaned_sentence)
        upper_words = target_words + max(60, int(target_words * 0.15))
        if (not allow_over) and total_words >= min_words and total_words + sentence_words > upper_words:
            return False
        if total_words + sentence_words > upper_words:
            return False
        if is_redundant(index, threshold=threshold):
            return False
        selected.append({"idx": int(index), "score": float(final_scores[index]), "textrank": float(base_scores[index]), "word_count": sentence_words, "text": raw_sentence, "clean_text": cleaned_sentence})
        selected_ids.append(index)
        total_words += sentence_words
        return True

    def best_index(pattern: str, pool: Optional[range] = None) -> Optional[int]:
        indexes = list(pool) if pool is not None else list(range(len(sentences)))
        candidates = [index for index in indexes if re.search(pattern, sentences[index], flags=re.IGNORECASE)]
        if not candidates:
            return None
        return max(candidates, key=lambda index: final_scores[index])

    total_sentences = len(sentences)
    early_pool = range(0, max(1, min(total_sentences, int(total_sentences * 0.28))))
    seed_patterns = [
        (r"mục tiêu|mục đích|nội dung chính|tóm tắt lại|tổng quan|đặt vấn đề", early_pool),
        (r"phương pháp|cách thức|quá trình|giai đoạn|thời kỳ|tiến trình", None),
        (r"tóm lại|nhìn chung|kết luận|kết quả|cho thấy|đánh giá", None),
        (r"số liệu|tỷ lệ|thống kê|thực thi|quản lý|tác động|ý nghĩa", None),
    ]

    for pattern, pool in seed_patterns:
        index = best_index(pattern, pool)
        if index is not None:
            add_index(index, allow_over=True, threshold=0.88)

    for index in order:
        add_index(index)
        if total_words >= min_words:
            break

    if total_words < min_words:
        for index in order:
            if add_index(index, allow_over=False, threshold=0.92):
                if total_words >= min_words:
                    break

    return sorted(selected, key=lambda item: item["idx"])


def textrank_extract(text: str, target_words: int = 500) -> tuple[list[dict], list[str], np.ndarray]:
    sentences = split_sentences(text)
    if len(sentences) < 5:
        raise RuntimeError(f"Không đủ câu để chạy TextRank: chỉ có {len(sentences)} câu.")
    matrix = build_tfidf_matrix(sentences)
    similarity = matrix @ matrix.T
    scores = pagerank_scores(similarity)
    selected = select_sentences_textrank(sentences=sentences, base_scores=scores, target_words=target_words, min_words=max(120, int(target_words * 0.90)))
    return selected, sentences, scores


def extractive_summary_from_selected(selected: list[dict]) -> str:
    cleaned = []
    seen = set()
    for item in selected:
        sentence = item.get("clean_text") or clean_sentence_for_display(item["text"])
        if not sentence:
            continue
        key = re.sub(r"\W+", " ", _strip_accents(sentence.lower())).strip()[:120]
        if key in seen:
            continue
        seen.add(key)
        cleaned.append(sentence)
    if len(cleaned) <= 4:
        return _squeeze(" ".join(cleaned))
    chunks = []
    step = max(2, math.ceil(len(cleaned) / 4))
    for index in range(0, len(cleaned), step):
        chunks.append(" ".join(cleaned[index:index + step]))
    return _squeeze("\n\n".join(chunks))


def _has_cjk(text: str) -> bool:
    return bool(re.search(r"[　-〿㐀-䶿一-鿿豈-﫿＀-￯\U00020000-\U0002a6df\U0002a700-\U0002ceaf]", text or ""))


_CJK_BAD_WORDS_IDS = None
_CJK_SCANNED = False


def get_cjk_bad_words_ids():
    global _CJK_BAD_WORDS_IDS, _CJK_SCANNED
    if _CJK_SCANNED:
        return _CJK_BAD_WORDS_IDS
    if sum_tokenizer is None:
        return None
    bad_ids = set()
    vocab = sum_tokenizer.get_vocab()
    for token, token_id in vocab.items():
        if _has_cjk(token):
            bad_ids.add(token_id)
    _CJK_SCANNED = True
    _CJK_BAD_WORDS_IDS = [[token_id] for token_id in sorted(bad_ids)] if bad_ids else None
    return _CJK_BAD_WORDS_IDS


def strip_bad_output(text: str) -> str:
    text = re.sub(r"<\|im_(start|end)\|>", "", text or "")
    text = re.sub(r"(?im)^\s*(system|user|assistant)\s*:?\s*$", "", text)
    text = re.sub(r"(?im)^\s*#{1,6}\s*", "", text)
    return _squeeze(text)


def qwen_polish_textrank(title: str, selected: list[dict], target_words: int = 500) -> str:
    if not ensure_sum_model() or sum_model is None or sum_tokenizer is None or torch is None:
        raise RuntimeError("Qwen summary model is unavailable in current environment")

    evidence = "\n".join(f"{index + 1}. {clean_sentence_for_display(item['text'])}" for index, item in enumerate(selected))
    system = (
        "Bạn là một nhà nghiên cứu chuyên viết báo cáo phân tích chi tiết. "
        "Nhiệm vụ của bạn là dựa trên các tư liệu lịch sử/khoa học được cung cấp, "
        "viết một bài tiểu luận hoàn chỉnh, diễn đạt trôi chảy và sâu sắc. "
        "Không viết tiếng Trung, không dùng định dạng danh sách (Markdown)."
    )
    user = f"""Hãy viết một bài tiểu luận phân tích CHI TIẾT có độ dài ít nhất {target_words} từ dựa trên các bằng chứng dưới đây.

YÊU CẦU BẮT BUỘC ĐỂ ĐẠT ĐỘ DÀI:
1. CẤU TRÚC: Bài viết PHẢI chia thành ít nhất 5 đoạn văn lớn. Mỗi đoạn phải dài từ 6 đến 8 câu.
2. PHÁT TRIỂN Ý: Đừng chỉ ghép nối các câu bằng chứng. Hãy sử dụng các từ nối, câu dẫn nhập (ví dụ: \"Trong bối cảnh đó...\", \"Bên cạnh đó...\", \"Đặc biệt đáng chú ý là...\") để giải thích rõ logic và bối cảnh của sự việc.
3. BẢO TOÀN CHI TIẾT: BẮT BUỘC phải đưa TOÀN BỘ tên các nhân vật, chức vụ, địa danh, và thời gian có trong bằng chứng vào bài viết một cách tự nhiên.
4. KẾT LUẬN SÂU SẮC: Đoạn cuối cùng phải đúc kết lại ý nghĩa của toàn bộ tài liệu, tuyệt đối không lặp lại nguyên văn các câu đã viết ở trên.
5. CẤM LẶP TỪ: TUYỆT ĐỐI KHÔNG sử dụng cấu trúc 'Điều này cho thấy...' hoặc 'Điều này chứng tỏ...'.
6. KHÔNG ĐƯỢC PHÉP XUẤT HIỆN TIẾNG ANH HOẶC TRUNG, HOẶC CÁC KÝ TỰ LẠ: Bài viết phải hoàn toàn bằng tiếng Việt.
TIÊU ĐỀ TÀI LIỆU:
{title}

CÁC BẰNG CHỨNG CỤ THỂ:
{evidence}

BÀI TIỂU LUẬN PHÂN TÍCH CHI TIẾT (Cam kết viết đủ {target_words} từ):"""
    prompt = f"<|im_start|>system\n{system}<|im_end|>\n<|im_start|>user\n{user}<|im_end|>\n<|im_start|>assistant\n"
    inputs = sum_tokenizer(prompt, return_tensors="pt", truncation=True, max_length=7600).to("cuda")
    kwargs = dict(
        **inputs,
        max_new_tokens=1500,
        do_sample=True,
        temperature=0.35,
        top_p=0.9,
        repetition_penalty=1.18,
        pad_token_id=sum_tokenizer.pad_token_id or sum_tokenizer.eos_token_id,
        eos_token_id=sum_tokenizer.eos_token_id,
    )
    bad_words_ids = get_cjk_bad_words_ids()
    if bad_words_ids:
        kwargs["bad_words_ids"] = bad_words_ids
    with torch.inference_mode():
        output = sum_model.generate(**kwargs)
    new_ids = output[0][inputs["input_ids"].shape[1]:]
    generated = sum_tokenizer.decode(new_ids, skip_special_tokens=True)
    return strip_bad_output(generated)


def looks_bad_polish(text: str, selected: list[dict], target_words: int) -> tuple[bool, list[str]]:
    reasons = []
    words = word_count(text)
    min_ok = max(80, int(target_words * 0.50))
    max_ok = int(target_words * 1.50)
    if words < min_ok:
        reasons.append(f"quá ngắn ({words}/{target_words} từ)")
    if words > max_ok:
        reasons.append(f"quá dài ({words}/{target_words} từ)")
    if re.search(r"(?m)^\s*(?:\d+\.|[-*])\s+", text):
        reasons.append("bị biến thành danh sách")
    return bool(reasons), reasons


@dataclass
class TextRankSummaryResult:
    title: str
    body_word_count: int
    sentence_count: int
    selected: list[dict]
    extractive: str
    final: str
    used_llm: bool
    used_fallback: bool
    guard_reasons: list[str]
    elapsed_seconds: float


def summarize_pdf_by_textrank(raw_text: str, target_words: int = 500, use_llm_polish: bool = True, verbose: bool = False) -> TextRankSummaryResult:
    started_at = time.time()
    title = extract_title(raw_text)
    body_text = clean_pdf_body_text(raw_text)

    if use_llm_polish:
        llm_compression_ratio = 0.45
        noise_margin = 0.15
        textrank_word_budget = int((target_words / llm_compression_ratio) * (1 + noise_margin))
    else:
        textrank_word_budget = target_words

    selected, sentences, _ = textrank_extract(body_text, target_words=textrank_word_budget)
    extractive = extractive_summary_from_selected(selected)

    final = extractive
    used_llm = False
    used_fallback = False
    guard_reasons: list[str] = []

    if use_llm_polish:
        try:
            candidate = qwen_polish_textrank(title, selected, target_words=target_words)
            bad, reasons = looks_bad_polish(candidate, selected, target_words)
            if bad:
                used_fallback = True
                guard_reasons = reasons
            else:
                final = candidate
                used_llm = True
        except Exception as error:
            used_fallback = True
            guard_reasons = [str(error)]

    return TextRankSummaryResult(
        title=title,
        body_word_count=word_count(body_text),
        sentence_count=len(sentences),
        selected=selected,
        extractive=extractive,
        final=_squeeze(final),
        used_llm=used_llm,
        used_fallback=used_fallback,
        guard_reasons=guard_reasons,
        elapsed_seconds=time.time() - started_at,
    )
