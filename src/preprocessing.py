"""
preprocessing.py
----------------
Normalizes raw contract text pulled from scraped SEC/EDGAR filings:
collapses whitespace/line-break artifacts, strips repeated page-header/footer
junk, and chunks long contracts so they fit an LLM context window with overlap
(so a clause split across a chunk boundary isn't lost).
"""

import re
from typing import List


def normalize_text(text: str) -> str:
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"\n\s*\d{1,4}\s*\n", "\n", text)
    return text.strip()


def chunk_text(text: str, max_chars: int = 12000, overlap: int = 800) -> List[str]:
    """
    Split a long contract into overlapping chunks so each fits comfortably in
    an LLM prompt alongside instructions. Most CUAD contracts are 5-15 pages
    (well under max_chars), so short contracts return a single chunk.
    """
    if len(text) <= max_chars:
        return [text]

    chunks = []
    start = 0
    while start < len(text):
        end = min(start + max_chars, len(text))
        chunks.append(text[start:end])
        if end == len(text):
            break
        start = end - overlap
    return chunks
