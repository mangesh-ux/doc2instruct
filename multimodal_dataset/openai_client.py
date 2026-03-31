from __future__ import annotations

import json
import time
from typing import Any

from openai import OpenAI


def _qa_json_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "page_status": {
                "type": "string",
                "enum": ["usable", "blank", "unreadable", "index_only", "image_only"],
            },
            "page_status_reason": {"type": "string"},
            "items": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "question": {"type": "string"},
                        "answer": {"type": "string"},
                        "question_type": {"type": "string"},
                        "difficulty": {"type": "string"},
                        "citation_quote": {"type": "string"},
                    },
                    "required": [
                        "question",
                        "answer",
                        "question_type",
                        "difficulty",
                        "citation_quote",
                    ],
                    "additionalProperties": False,
                },
            }
        },
        "required": ["page_status", "page_status_reason", "items"],
        "additionalProperties": False,
    }


def _read_field(obj: Any, key: str, default: int = 0) -> int:
    if obj is None:
        return default
    if isinstance(obj, dict):
        return int(obj.get(key, default) or default)
    return int(getattr(obj, key, default) or default)


def _extract_usage(response: Any) -> tuple[int, int]:
    usage = getattr(response, "usage", None)
    return _read_field(usage, "input_tokens"), _read_field(usage, "output_tokens")


def _estimate_cost_usd(
    *,
    input_tokens: int,
    output_tokens: int,
    input_cost_per_1m_tokens_usd: float,
    output_cost_per_1m_tokens_usd: float,
) -> float:
    return (
        (input_tokens / 1_000_000.0) * input_cost_per_1m_tokens_usd
        + (output_tokens / 1_000_000.0) * output_cost_per_1m_tokens_usd
    )


def generate_qa_batch(
    *,
    client: OpenAI,
    model: str,
    temperature: float,
    system_prompt: str,
    user_prompt: str,
    image_data_url: str,
    input_cost_per_1m_tokens_usd: float,
    output_cost_per_1m_tokens_usd: float,
) -> tuple[dict[str, Any], dict[str, Any]]:
    started = time.perf_counter()
    response = client.responses.create(
        model=model,
        temperature=temperature,
        input=[
            {"role": "system", "content": [{"type": "input_text", "text": system_prompt}]},
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
                "name": "qa_batch",
                "schema": _qa_json_schema(),
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
