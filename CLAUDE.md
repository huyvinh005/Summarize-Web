# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository status

This repository is currently an early-stage workspace, not yet a conventional app codebase. The committed project contents are:
- `backend/dataops-aisumary-9.ipynb`: the main PDF ingestion and summarization research notebook
- `backend/RAG.ipynb`: a separate retrieval-augmented generation notebook
- `frontend/resend.com_.png`: a visual reference for the intended frontend design direction

There is currently no checked-in web app scaffold, package manifest, test suite, or build configuration. If future work adds an actual frontend or backend app, update this file to document the new runtime commands and folder responsibilities.

## Common commands

Because the repo currently consists of notebooks rather than a packaged application, there are no canonical build/lint/test commands in the repository yet.

### Current development workflow
- Open and run the notebooks interactively:
  - `backend/dataops-aisumary-9.ipynb`
  - `backend/RAG.ipynb`
- If working locally outside Kaggle/Colab, export notebook cells into Python modules before trying to introduce CI, linting, or automated tests.

### Commands embedded in notebooks
These are the environment setup commands currently used inside notebook cells and should be treated as the closest thing to source-of-truth dependencies:

For PDF extraction / OCR in `backend/dataops-aisumary-9.ipynb`:
- `pip install PyMuPDF pdfplumber pdf2image pytesseract`
- `apt-get install -y tesseract-ocr tesseract-ocr-vie poppler-utils`
- `pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"`
- `pip install --no-deps xformers "trl<0.9.0" peft accelerate bitsandbytes`

For retrieval work in `backend/RAG.ipynb`:
- `pip install langchain langchain-community langchain-huggingface sentence-transformers chromadb`

### Important note for future app work
If you add a real web application, add the exact commands for:
- install/dependency sync
- dev server(s)
- production build
- lint/typecheck
- test suite and single-test execution

Do not guess those commands from ecosystem defaults; derive them from committed package/config files.

## Mandatory implementation rules

### Backend
- The summarization model and summarization workflow must follow `backend/dataops-aisumary-9.ipynb`.
- The AI chat / user interaction workflow must follow `backend/RAG.ipynb`.
- When implementing production backend code, treat those two notebooks as the authoritative product logic for their respective features.
- Do not swap responsibilities between the two notebooks: summarization belongs to `dataops-aisumary-9.ipynb`, and article chat / RAG belongs to `RAG.ipynb`.

### Frontend
- The website must be mobile-friendly.
- Every major section should include scroll-triggered animation.
- Keep the visual direction aligned with the clean SaaS style implied by `frontend/resend.com_.png`.

## High-level architecture

The current repository has two mostly independent backend research tracks that should be understood before turning this into a production website.

### 1. PDF ingestion and summary pipeline (`backend/dataops-aisumary-9.ipynb`)

This notebook is the most complete expression of the current product logic. It is structured as a staged pipeline:

1. **PDF classification before extraction**
   - Uses `PyMuPDF` (`fitz`) to inspect each PDF.
   - Computes signals such as average characters per page, text-page ratio, image coverage, missing embedded fonts, garbage-character ratio, and legacy Vietnamese font encoding detection.
   - Decides whether a file can be read directly or must go through OCR.

2. **Dual extraction path**
   - **Direct extraction path** uses `pdfplumber`/`PyMuPDF` for text-first PDFs.
   - Includes custom handling for **two-column academic layouts** by detecting a central whitespace gap and extracting left/right columns separately.
   - **OCR path** rasterizes pages with `pdf2image` and runs `pytesseract` with Vietnamese + English OCR.

3. **Normalization and cleanup**
   - The notebook contains substantial Vietnamese-oriented text cleanup logic.
   - It strips noise such as abstracts, references, publication metadata, page artifacts, malformed Unicode, and repeated boilerplate.
   - This cleanup is not generic utility code yet; it is embedded directly in notebook functions.

4. **Extractive ranking / candidate selection**
   - The notebook chunks long text into overlapping word windows.
   - A fine-tuned local Qwen model scores chunks for importance.
   - There is also a separate TextRank-style sentence graph pipeline using TF-IDF similarity and PageRank-style scoring.
   - The notebook is effectively experimenting with a hybrid strategy: heuristic filtering + model scoring + extractive selection.

5. **Abstractive polishing**
   - After extractive selection, a larger Qwen model is used to rewrite or polish the chosen evidence into a smoother Vietnamese summary.
   - Guardrails are included to detect outputs that are too short, too list-like, or otherwise low quality, then fall back to the extractive summary.

### 2. RAG prototype (`backend/RAG.ipynb`)

This notebook is a separate retrieval-focused prototype and should be treated as a second architecture track rather than part of the same pipeline.

It currently does the following:
- Uses `langchain_huggingface.HuggingFaceEmbeddings`
- Stores vectors in `Chroma`
- Performs `similarity_search` and also creates a retriever via `vectorstore.as_retriever(...)`
- Loads a local Qwen instruct model for answer generation
- Builds a LangChain-style prompt/retrieval chain around retrieved context

This means the repo already contains two distinct product directions:
- **document summarization pipeline**
- **question-answering over retrieved document chunks**

For the intended website, these map naturally to:
- the main summary output panel
- the right-side “chat with the article” panel

## Product mapping for future web implementation

The intended website described for this repository should be built around the existing notebook logic, not as a greenfield AI app with unrelated assumptions.

### Recommended feature decomposition

- **Authentication layer**
  - Login, registration, and email verification are product requirements, but no auth code exists yet.
  - This is currently separate from the research logic in the notebooks.

- **Document ingestion service**
  - Should own PDF upload, OCR decisioning, text extraction, and normalized document storage.
  - The notebook extraction logic is the best current source for this service’s behavior.

- **Summary generation service**
  - Should own chunking, ranking, extractive summary generation, and optional abstractive polishing.
  - The notebook mixes all of this in one place today, so future refactors should split these responsibilities explicitly.

- **RAG / article chat service**
  - Should own embeddings, vector storage, retrieval, and question answering over the article.
  - `backend/RAG.ipynb` is the best current reference for this behavior.

- **Dashboard application**
  - Should expose four main product surfaces:
    - summary history
    - text/PDF input
    - primary summary output
    - contextual AI chat about the current article

## Important implementation constraints derived from the notebooks

### OCR and direct extraction are both first-class
Do not assume all PDFs should be OCR’d. The current notebook explicitly invests in classification so direct extraction can be used when reliable, with OCR as fallback for scanned or badly encoded documents.

### Vietnamese text handling is a core requirement
A lot of the notebook complexity exists because Vietnamese PDFs can fail in non-obvious ways:
- legacy TCVN3/VNI/ABC-like encodings
- malformed Unicode extraction
- OCR language requirements (`vie+eng`)
- noisy academic formatting

If you port notebook logic into application code, preserve these behaviors rather than replacing them with a generic PDF parser.

### Two-column academic PDFs matter
The direct extraction path includes explicit two-column detection and column-wise reading order reconstruction. That is a meaningful part of summary quality and should not be dropped casually.

### The current summarization approach is hybrid, not a single LLM call
The notebooks do not simply “upload PDF -> ask model to summarize.” They use:
- document inspection
- extraction path selection
- cleanup
- chunking / ranking
- extractive selection
- optional polishing

Future code should preserve that staged architecture unless there is a deliberate, validated simplification.

### RAG notebook uses deprecated Chroma import paths
`backend/RAG.ipynb` still uses `langchain_community.vectorstores.Chroma`, and notebook output already shows a deprecation warning recommending `langchain-chroma`. If productionizing the RAG path, update these imports first instead of copying the deprecated pattern.

## Frontend design direction

The only checked-in frontend artifact is `frontend/resend.com_.png`, which serves as a style reference rather than source code. The expected design direction is:
- minimalist SaaS layout
- generous whitespace
- restrained typography
- clean card surfaces with subtle borders/radius
- professional, modern dashboard styling rather than a flashy AI-chat aesthetic

If future frontend code is added, keep the summary workspace feeling more like a focused document tool than a general chatbot landing page.

## What future Claude instances should assume

- This repo is currently **research-first**, not **app-first**.
- The notebooks are the authoritative expression of the product’s AI behavior.
- If you are asked to implement the website, first extract reusable Python/backend modules from notebook logic instead of rebuilding the algorithms from scratch.
- If package manifests, test runners, or app scaffolds are added later, this file should be updated immediately with concrete commands and the new module boundaries.
