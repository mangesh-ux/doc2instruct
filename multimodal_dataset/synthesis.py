"""Cross-page synthesis API wrapper and strict output schema."""

from __future__ import annotations

import json
import time
from typing import Any

from openai import OpenAI

from multimodal_dataset.openai_client import _estimate_cost_usd, _extract_usage


def _synthesis_schema(max_evidence_quotes_per_item: int) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "pack_status": {
                "type": "string",
                "enum": ["usable", "insufficient_evidence", "redundant", "unrelated"],
            },
            "pack_status_reason": {"type": "string"},
            "items": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "question": {"type": "string"},
                        "answer": {"type": "string"},
                        "question_type": {"type": "string"},
                        "difficulty": {"type": "string"},
                        "requires_multi_page_reasoning": {"type": "boolean"},
                        "source_pages": {
                            "type": "array",
                            "items": {"type": "integer"},
                            "minItems": 2,
                        },
                        "evidence_quotes": {
                            "type": "array",
                            "maxItems": max_evidence_quotes_per_item,
                            "items": {
                                "type": "object",
                                "properties": {
                                    "page": {"type": "integer"},
                                    "quote": {"type": "string"},
                                },
                                "required": ["page", "quote"],
                                "additionalProperties": False,
                            },
                        },
                        "synthesis_type": {
                            "type": "string",
                            "enum": [
                                "section_synthesis",
                                "cross_page_multi_hop",
                                "compare_contrast",
                                "prerequisite_application",
                                "chapter_progression",
                                "concept_unification",
                            ],
                        },
                    },
                    "required": [
                        "question",
                        "answer",
                        "question_type",
                        "difficulty",
                        "requires_multi_page_reasoning",
                        "source_pages",
                        "evidence_quotes",
                        "synthesis_type",
                    ],
                    "additionalProperties": False,
                },
            },
        },
        "required": ["pack_status", "pack_status_reason", "items"],
        "additionalProperties": False,
    }


def synthesize_cross_page_batch(
    *,
    client: OpenAI,
    model: str,
    temperature: float,
    evidence_pack: dict[str, Any],
    max_cross_page_qas_per_pack: int,
    max_evidence_quotes_per_item: int,
    use_local_qas_as_hints: bool,
    input_cost_per_1m_tokens_usd: float,
    output_cost_per_1m_tokens_usd: float,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Generate cross-page QA from an evidence pack with strict schema."""
    page_text_lines = [
        f"Page {p['page']}:\n{p['text'][:4000]}" for p in evidence_pack.get("page_texts", [])
    ]
    hints = evidence_pack.get("accepted_local_qas", [])[:8] if use_local_qas_as_hints else []
    user_prompt = (
        "You are synthesizing cross-page instruction data from source evidence.\n"
        "Source evidence is the ground truth. Local QAs are hints only.\n"
        "Generate only questions that require combining multiple pages.\n"
        "If evidence is insufficient/unrelated/redundant, return zero items and set pack_status.\n"
        f"Generate at most {max_cross_page_qas_per_pack} items.\n"
        f"Evidence pack id: {evidence_pack.get('pack_id')}\n"
        f"Source pages: {evidence_pack.get('source_pages')}\n"
        f"Shared terms: {evidence_pack.get('shared_terms')}\n"
        f"Heading candidates: {evidence_pack.get('heading_candidates')}\n"
        "Page texts:\n"
        + "\n\n".join(page_text_lines)
        + "\n\nLocal QA hints:\n"
        + json.dumps(hints, ensure_ascii=False)
    )
    system_prompt = (
        "Return strict JSON. Every accepted item must depend on multiple pages and include "
        "traceable evidence quotes from those pages."
    )

    started = time.perf_counter()
    response = client.responses.create(
        model=model,
        temperature=temperature,
        input=[
            {"role": "system", "content": [{"type": "input_text", "text": system_prompt}]},
            {"role": "user", "content": [{"type": "input_text", "text": user_prompt}]},
        ],
        text={
            "format": {
                "type": "json_schema",
                "name": "cross_page_synthesis",
                "schema": _synthesis_schema(max_evidence_quotes_per_item),
                "strict": True,
            }
        },
    )
    latency_ms = int((time.perf_counter() - started) * 1000)
    input_tokens, output_tokens = _extract_usage(response)
    estimated_cost = _estimate_cost_usd(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        input_cost_per_1m_tokens_usd=input_cost_per_1m_tokens_usd,
        output_cost_per_1m_tokens_usd=output_cost_per_1m_tokens_usd,
    )
    payload = json.loads(response.output_text)
    metrics = {
        "model": model,
        "latency_ms": latency_ms,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "estimated_cost_usd": round(estimated_cost, 8),
    }
    return payload, metrics
