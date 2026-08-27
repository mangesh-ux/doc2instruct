"""Quality and dedup helpers plus model-based critique call wrapper."""

from __future__ import annotations

import difflib
import json
import re
import time
import unicodedata
from typing import Any

from openai import OpenAI
from multimodal_dataset.openai_client import _estimate_cost_usd, _extract_usage

# Fraction of a citation quote that must appear as one contiguous run in the
# page text for the citation to count as grounded.
#
# This is deliberately below 1.0. The model transcribes quotes from the page
# *image* while we match against *PyMuPDF-extracted* text, and the two
# disagree on ligatures, soft hyphens, column breaks and maths spacing even
# when the quote is a faithful transcription. Requiring an exact substring
# rejects almost every quote on born-digital papers (measured: 31/40 local
# items and 6/6 cross-page items rejected, while the judge scored those same
# items 1.0 for grounding).
MIN_QUOTE_COVERAGE = 0.85
MIN_QUOTE_CHARS = 8
# Fragments shorter than this are dropped rather than required to match; they
# are usually punctuation debris left behind after splitting on an ellipsis.
MIN_FRAGMENT_CHARS = 12
# Models routinely elide material inside a citation ("first clause ... later
# clause"). Such a quote is still faithful, but it is not one contiguous span,
# so it must be matched fragment by fragment.
_ELLIPSIS_SPLIT = re.compile(r"\s*(?:\.\s*\.\s*\.+|\u2026)\s*")


def normalize_text(value: str) -> str:
    """Normalize whitespace/case for robust similarity checks."""
    lowered = value.lower().strip()
    return re.sub(r"\s+", " ", lowered)


def normalize_for_match(value: str) -> str:
    """Normalize text for verbatim quote matching against PDF-extracted text.

    Beyond case/whitespace this folds unicode compatibility forms (ligatures
    like "fi", full-width chars), rejoins words split across line breaks
    ("repre-\\nsentation" -> "representation"), and unifies curly quotes and
    dashes. These are the differences that make a faithful quote fail a naive
    substring test.
    """
    text = unicodedata.normalize("NFKC", value)
    text = re.sub(r"-\s*\n\s*", "", text)
    text = text.replace("\u2019", "'").replace("\u2018", "'")
    text = text.replace("\u201c", '"').replace("\u201d", '"')
    text = text.replace("\u2013", "-").replace("\u2014", "-")
    text = re.sub(r"\s+", " ", text)
    return text.strip().lower()


def text_similarity(a: str, b: str) -> float:
    """Return approximate text similarity in [0, 1]."""
    return difflib.SequenceMatcher(a=normalize_text(a), b=normalize_text(b)).ratio()


def quote_coverage(source_text: str, quote: str) -> float:
    """Fraction of `quote` present as one contiguous run inside `source_text`.

    Returns 1.0 on exact containment. Otherwise falls back to the longest
    common substring, so a single stray extraction artefact costs a little
    coverage instead of failing the quote outright.
    """
    source_norm = normalize_for_match(source_text)
    quote_norm = normalize_for_match(quote)
    if not source_norm or not quote_norm:
        return 0.0
    if quote_norm in source_norm:
        return 1.0
    matcher = difflib.SequenceMatcher(a=quote_norm, b=source_norm, autojunk=False)
    match = matcher.find_longest_match(0, len(quote_norm), 0, len(source_norm))
    return match.size / len(quote_norm)


def quote_fragments(citation_quote: str) -> list[str]:
    """Split a citation on ellipses into the spans that must each be verbatim."""
    parts = [p.strip() for p in _ELLIPSIS_SPLIT.split(citation_quote or "")]
    fragments = [p for p in parts if len(normalize_for_match(p)) >= MIN_FRAGMENT_CHARS]
    # A quote with no fragment long enough to be meaningful falls back to the
    # whole string, so short single-span quotes still get checked.
    if not fragments:
        whole = (citation_quote or "").strip()
        return [whole] if whole else []
    return fragments


def citation_coverage(page_text: str, citation_quote: str) -> float:
    """Coverage of the weakest fragment of a citation quote.

    Reporting the worst fragment (rather than the mean) keeps the metric
    conservative: a citation is only as grounded as its least-supported part.
    """
    fragments = quote_fragments(citation_quote)
    if not fragments:
        return 0.0
    return min(quote_coverage(page_text, fragment) for fragment in fragments)


def has_citation_match(
    page_text: str,
    citation_quote: str,
    min_coverage: float = MIN_QUOTE_COVERAGE,
) -> bool:
    """Check whether citation_quote is supported by the extracted page text.

    Every ellipsis-separated fragment must independently appear on the page, so
    an elided quote counts as grounded while a fabricated one does not.
    """
    if not page_text.strip() or not citation_quote.strip():
        return False
    if len(normalize_for_match(citation_quote)) < MIN_QUOTE_CHARS:
        return False
    return citation_coverage(page_text, citation_quote) >= min_coverage


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
