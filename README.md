# AI ATS Resume Analyzer — RAG + Safe LaTeX Tailoring

A local-first portfolio project that analyzes a resume against a job description using the RAG pipeline implemented in `AI_ATS_RAG_Resume_Analyzer_v2.ipynb`, then optionally generates a tailored LaTeX resume using the supplied design language.

## What it does

**Phase 1 — RAG ATS Analyzer**

1. Accepts a resume PDF and complete JD.
2. Extracts PDF text with `pypdf` and cleans whitespace.
3. Chunks both sources using the notebook's custom **180-word chunks / 40-word overlap**.
4. Creates normalized embeddings with `sentence-transformers/all-MiniLM-L6-v2`.
5. Indexes resume and JD chunks in in-memory Chroma with source metadata.
6. Performs multi-query retrieval for JD and resume evidence.
7. Reranks candidates with cosine similarity.
8. Uses the LLM to dynamically extract JD requirements and the candidate profile from the actual evidence.
9. Builds a keyword set from the actual JD rather than a hardcoded skills dictionary.
10. Performs exact and semantic matching with a default 0.72 cosine threshold.
11. Generates structured matched, partial, missing, keyword, project, experience, education, issue, recommendation and evidence outputs.

**Phase 2 — Resume Tailoring**

The RAG result controls a LaTeX tailoring pass. The generator can rewrite and reorder truthful content, but it must not invent technologies, projects, experience, certifications, grades, achievements or responsibilities.

## Run locally

### 1. Create environment

```bash
python -m venv .venv
# Windows
.venv\\Scripts\\activate
# macOS/Linux
source .venv/bin/activate
```

### 2. Install

```bash
pip install -r requirements.txt
```

### 3. Configure API key

Copy `.env.example` to `.env`.

For Hugging Face:

```env
LLM_PROVIDER=huggingface
HF_TOKEN=your_token
HF_LLM_MODEL=openai/gpt-oss-120b
```

If your selected Hugging Face provider/model is unavailable or quota-limited, switch to another legitimate provider/model supported by your account, or use the OpenAI provider configuration.

**Never commit `.env` or a real API token.**

### 4. Start the app

```bash
streamlit run app/main.py
```

Open the local Streamlit URL shown in the terminal.

## LaTeX PDF generation

The app always generates a `.tex` file when tailoring succeeds. If `pdflatex` is installed and available on PATH, the app also compiles `tailored_resume.pdf` automatically. If not, the app explains that the `.tex` file is ready and can be compiled after installing MiKTeX or TeX Live.

## Project structure

```text
AI_ATS_Resume_Analyzer/
├── app/main.py
├── services/
│   ├── analyzer.py
│   ├── config.py
│   ├── embeddings.py
│   ├── latex.py
│   ├── llm.py
│   ├── pdf_utils.py
│   ├── prompts.py
│   ├── retrieval.py
│   └── vector_store.py
├── templates/resume_template.tex
├── tests/test_core.py
├── outputs/
├── data/
├── .env.example
├── .gitignore
├── requirements.txt
└── README.md
```

## Architecture

```text
Resume PDF + JD
      ↓
Extraction + Cleaning
      ↓
Custom 180/40 Chunking
      ↓
Local Sentence-Transformer Embeddings
      ↓
Chroma Index + Metadata
      ↓
Multi-query Retrieval
      ↓
Cosine Similarity Reranking
      ↓
Retrieved Evidence
      ↓
Dynamic JD Profile + Dynamic Resume Profile
      ↓
Exact / Semantic Matching
      ↓
Structured RAG ATS Analysis
      ↓
Safe Resume Tailoring
      ↓
LaTeX → optional PDF
```

## Important metric note

The displayed “JD Coverage” and “Semantic Coverage” are **project-specific coverage metrics** based on extracted JD terms. They are not a claim about the proprietary scoring formula used by an employer's ATS.

## Security

- API keys are read from `.env`.
- Real tokens are never included in the repository.
- Uploaded resumes are processed locally during the session.
- Embeddings and Chroma indexing run locally.
- Only LLM generation is sent to the configured API provider.

## Tests

```bash
pytest -q
```

## Interview explanation

A concise explanation is:

> “I built a RAG-based ATS resume analyzer. I extract and clean the resume PDF and JD, split them into overlapping chunks, create local sentence-transformer embeddings, and store them in Chroma with source metadata. For retrieval I use multiple requirement-focused queries and rerank the retrieved chunks using cosine similarity. Then I augment an LLM prompt with the retrieved evidence and dynamically extract both the JD requirements and candidate profile. Finally I perform exact and semantic matching and generate structured ATS insights. In Phase 2, those insights drive a constrained LaTeX resume generator that improves alignment without fabricating candidate information.”

## Source notebook

`AI_ATS_RAG_Resume_Analyzer_v2_sanitized.ipynb` is included as a sanitized source-of-truth reference. Its authentication token has been removed; configure credentials through `.env` instead.
