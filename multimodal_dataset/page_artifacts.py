"""Stage-1 page artifact creation and lightweight signal extraction."""

from __future__ import annotations

import re
from collections import Counter
from typing import Any


_STOPWORDS = {
    "the",
    "and",
    "for",
    "that",
    "with",
    "this",
    "from",
    "are",
    "was",
    "were",
    "have",
    "has",
    "had",
    "about",
    "into",
    "your",
    "their",
    "what",
    "when",
    "where",
    "which",
    "while",
    "would",
    "could",
    "should",
}


def normalize_terms(text: str) -> list[str]:
    """Normalize text into lightweight lexical terms for overlap checks."""
    tokens = re.findall(r"[a-zA-Z]{3,}", text.lower())
    return [t for t in tokens if t not in _STOPWORDS]


def heading_candidate(page_text: str) -> str:
    """Pick a simple heading candidate from the first non-empty lines."""
    for line in page_text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if len(stripped) > 120:
            continue
        return stripped
    return ""


def first_non_empty_lines(page_text: str, limit: int = 3) -> list[str]:
    """Return first short non-empty lines for page preview."""
    lines: list[str] = []
    for line in page_text.splitlines():
        stripped = line.strip()
        if stripped:
            lines.append(stripped[:200])
        if len(lines) >= limit:
            break
    return lines


def build_page_artifact(
    *,
    run_id: str,
    source_book: str,
    source_page: int,
    page_status: str,
    page_status_reason: str,
    page_text: str,
    accepted_qas: list[dict[str, Any]],
) -> dict[str, Any]:
    """Build the stage-1 durable artifact for one processed page."""
    joined_qa_text = " ".join(
        f"{qa.get('question', '')} {qa.get('answer', '')}" for qa in accepted_qas
    )
    terms = normalize_terms(f"{page_text}\n{joined_qa_text}")
    term_counts = Counter(terms)
    top_terms = [term for term, _ in term_counts.most_common(20)]
    qtype_counts = Counter(str(qa.get("question_type", "unknown")) for qa in accepted_qas)

    return {
        "run_id": run_id,
        "source_book": source_book,
        "source_page": source_page,
        "page_status": page_status,
        "page_status_reason": page_status_reason,
        "page_text": page_text,
        "accepted_qas": accepted_qas,
        "heading_candidate": heading_candidate(page_text),
        "first_non_empty_lines": first_non_empty_lines(page_text),
        "keywords": top_terms,
        "question_type_distribution": dict(qtype_counts),
    }
