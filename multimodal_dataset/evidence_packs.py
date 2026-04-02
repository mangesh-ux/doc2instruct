"""Deterministic evidence-pack construction for cross-page synthesis."""

from __future__ import annotations

import itertools
import re
from typing import Any


def _normalize_heading(value: str) -> str:
    return re.sub(r"\s+", " ", value.lower().strip())


def _shared_terms(artifacts: list[dict[str, Any]], max_terms: int = 20) -> list[str]:
    term_sets = [set(a.get("keywords", [])) for a in artifacts if a.get("keywords")]
    if not term_sets:
        return []
    overlap = set.intersection(*term_sets)
    return sorted(list(overlap))[:max_terms]


def build_evidence_packs_for_book(
    *,
    source_book: str,
    page_artifacts: list[dict[str, Any]],
    min_pages_per_pack: int,
    max_pages_per_pack: int,
    pack_overlap_window: int,
) -> list[dict[str, Any]]:
    """Build packs from adjacent windows, lexical overlap, and heading continuity."""
    usable = [
        a for a in page_artifacts if a.get("page_status") == "usable" and a.get("accepted_qas")
    ]
    usable.sort(key=lambda x: int(x.get("source_page", 0)))
    if not usable:
        return []

    max_pages = max(min_pages_per_pack, max_pages_per_pack)
    step = max(1, max_pages - max(0, pack_overlap_window))
    packs: list[dict[str, Any]] = []

    for start in range(0, len(usable), step):
        window = usable[start : start + max_pages]
        if len(window) < min_pages_per_pack:
            continue

        pages = [int(a["source_page"]) for a in window]
        heading_candidates = [str(a.get("heading_candidate", "")) for a in window]
        normalized_headings = [_normalize_heading(h) for h in heading_candidates if h]
        heading_continuity = len(set(normalized_headings)) < len(normalized_headings)
        shared_terms = _shared_terms(window)
        strategy_parts = ["adjacent_window"]
        if shared_terms:
            strategy_parts.append("lexical_overlap")
        if heading_continuity:
            strategy_parts.append("heading_continuity")

        evidence_snippets = list(
            itertools.chain.from_iterable(a.get("first_non_empty_lines", [])[:2] for a in window)
        )[:8]
        packs.append(
            {
                "source_book": source_book,
                "source_pages": pages,
                "pack_id": f"{source_book}::{'-'.join(str(p) for p in pages)}",
                "pack_strategy": "+".join(strategy_parts),
                "shared_terms": shared_terms,
                "heading_candidates": heading_candidates,
                "page_texts": [
                    {"page": int(a["source_page"]), "text": str(a.get("page_text", ""))}
                    for a in window
                ],
                "accepted_local_qas": list(
                    itertools.chain.from_iterable(a.get("accepted_qas", []) for a in window)
                ),
                "evidence_snippets": evidence_snippets,
            }
        )

    return packs
