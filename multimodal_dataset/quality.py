"""Quality and dedup helpers plus model-based critique call wrapper."""

from __future__ import annotations

import difflib
import json
import re
import time
from typing import Any

from openai import OpenAI
from multimodal_dataset.openai_client import _estimate_cost_usd, _extract_usage


def normalize_text(value: str) -> str:
    """Normalize whitespace/case for robust similarity checks."""
    lowered = value.lower().strip()
    return re.sub(r"\s+", " ", lowered)


def text_similarity(a: str, b: str) -> float:
    """Return approximate text similarity in [0, 1]."""
    return difflib.SequenceMatcher(a=normalize_text(a), b=normalize_text(b)).ratio()


def has_citation_match(page_text: str, citation_quote: str) -> bool:
    """Check whether citation_quote exists in extracted page text."""
    if not page_text.strip() or not citation_quote.strip():
        return False
    page_norm = normalize_text(page_text)
    quote_norm = normalize_text(citation_quote)
    if len(quote_norm) < 8:
        return False
    return quote_norm in page_norm


def heuristic_usefulness_score(question: str, answer: str) -> float:
    """Fast pre-score that penalizes trivial/underspecified QA pairs."""
    question_words = len(question.split())
    answer_words = len(answer.split())
    if question_words < 4 or answer_words < 8:
        return 0.25
    if question_words < 6 or answer_words < 15:
        return 0.5
    if answer_words > 120:
        return 0.7
    return 0.85


def critique_qa_item(
    *,
    client: OpenAI,
    model: str,
    system_prompt: str,
    image_data_url: str,
    qa_item: dict[str, Any],
    input_cost_per_1m_tokens_usd: float,
    output_cost_per_1m_tokens_usd: float,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Run judge-model critique and return verdict + telemetry metrics."""
    critique_schema = {
        "type": "object",
        "properties": {
            "grounding_score": {"type": "number"},
            "usefulness_score": {"type": "number"},
            "grounded": {"type": "boolean"},
            "useful": {"type": "boolean"},
            "concerns": {
                "type": "array",
                "items": {"type": "string"},
            },
        },
        "required": [
            "grounding_score",
            "usefulness_score",
            "grounded",
            "useful",
            "concerns",
        ],
        "additionalProperties": False,
    }

    user_prompt = (
        "Critique this generated QA item against the page image.\n"
        "Score grounding_score and usefulness_score between 0 and 1.\n"
        "grounded=true only if answer is supported by visible page content.\n"
        "useful=true only if the question-answer pair would help instruction fine-tuning.\n"
        f"QA JSON:\n{json.dumps(qa_item, ensure_ascii=False)}"
    )

    started = time.perf_counter()
    response = client.responses.create(
        model=model,
        temperature=0,
        input=[
            {
                "role": "system",
                "content": [{"type": "input_text", "text": system_prompt}],
            },
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": user_prompt},
                    {"type": "input_image", "image_url": image_data_url},
                ],
            },
        ],
        text={
            "format": {
                "type": "json_schema",
                "name": "qa_critique",
                "schema": critique_schema,
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
    metrics = {
        "model": model,
        "latency_ms": latency_ms,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "estimated_cost_usd": round(estimated_cost, 8),
    }
    return json.loads(response.output_text), metrics


def critique_cross_page_item(
    *,
    client: OpenAI,
    model: str,
    evidence_pack: dict[str, Any],
    item: dict[str, Any],
    input_cost_per_1m_tokens_usd: float,
    output_cost_per_1m_tokens_usd: float,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Critique cross-page item for grounding, usefulness, and multi-page validity."""
    critique_schema = {
        "type": "object",
        "properties": {
            "grounding_score": {"type": "number"},
            "usefulness_score": {"type": "number"},
            "multi_page_score": {"type": "number"},
            "grounded": {"type": "boolean"},
            "useful": {"type": "boolean"},
            "truly_multi_page": {"type": "boolean"},
            "concerns": {"type": "array", "items": {"type": "string"}},
        },
        "required": [
            "grounding_score",
            "usefulness_score",
            "multi_page_score",
            "grounded",
            "useful",
            "truly_multi_page",
            "concerns",
        ],
        "additionalProperties": False,
    }
    page_texts = [
        {"page": p.get("page"), "text": str(p.get("text", ""))[:2000]}
        for p in evidence_pack.get("page_texts", [])
    ]
    user_prompt = (
        "Critique this cross-page QA item using only the provided evidence pack.\n"
        "Score in [0,1]. truly_multi_page=true only if answer requires multiple pages.\n"
        f"Evidence pack:\n{json.dumps(page_texts, ensure_ascii=False)}\n"
        f"Item:\n{json.dumps(item, ensure_ascii=False)}"
    )
    started = time.perf_counter()
    response = client.responses.create(
        model=model,
        temperature=0,
        input=[
            {
                "role": "system",
                "content": [
                    {
                        "type": "input_text",
                        "text": "You are a strict cross-page dataset quality auditor.",
                    }
                ],
            },
            {"role": "user", "content": [{"type": "input_text", "text": user_prompt}]},
        ],
        text={
            "format": {
                "type": "json_schema",
                "name": "cross_page_critique",
                "schema": critique_schema,
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
    metrics = {
        "model": model,
        "latency_ms": latency_ms,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "estimated_cost_usd": round(estimated_cost, 8),
    }
    return json.loads(response.output_text), metrics
