from __future__ import annotations

import json
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


def generate_qa_batch(
    *,
    client: OpenAI,
    model: str,
    temperature: float,
    system_prompt: str,
    user_prompt: str,
    image_data_url: str,
) -> dict[str, Any]:
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
    return json.loads(response.output_text)
