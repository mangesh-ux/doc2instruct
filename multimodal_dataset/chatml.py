"""ChatML record formatting and durable JSONL writing utilities."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any


def qa_to_chatml_record(
    *,
    qa_item: dict[str, Any],
    book_name: str,
    page_number: int,
    user_profile: str,
) -> dict[str, Any]:
    """Convert one QA item into the project's ChatML JSON record format."""
    system_msg = "You are a helpful tutor grounded in source material."
    if user_profile:
        system_msg += f" Tailor style for this user profile: {user_profile}"

    user_msg = qa_item["question"]
    assistant_msg = qa_item["answer"]

    return {
        "messages": [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg},
            {"role": "assistant", "content": assistant_msg},
        ],
        "metadata": {
            "source_book": book_name,
            "source_page": page_number,
            "question_type": qa_item.get("question_type", ""),
            "difficulty": qa_item.get("difficulty", ""),
            "citation_quote": qa_item.get("citation_quote", ""),
        },
    }


def append_jsonl(path: Path, record: dict[str, Any]) -> None:
    """Append one JSON line and fsync to reduce data-loss risk."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())
