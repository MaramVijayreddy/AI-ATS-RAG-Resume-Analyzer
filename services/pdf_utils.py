import re
from pathlib import Path
from pypdf import PdfReader

def extract_pdf_text(path: str | Path) -> str:
    reader = PdfReader(str(path))
    pages = []
    for page_no, page in enumerate(reader.pages, 1):
        text = page.extract_text() or ''
        if text.strip():
            pages.append(f'[PAGE {page_no}]\n{text}')
    return '\n'.join(pages)

def clean_text(text: str) -> str:
    return re.sub(r'\s+', ' ', text.replace('\x00',' ')).strip()

def chunk_text(text: str, chunk_size: int = 180, overlap: int = 40) -> list[str]:
    words = text.split()
    if not words: return []
    chunks=[]; start=0
    while start < len(words):
        end=min(start+chunk_size,len(words))
        chunks.append(' '.join(words[start:end]))
        if end == len(words): break
        start=max(end-overlap,start+1)
    return chunks
