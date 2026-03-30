from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass
class InputConfig:
    books_dir: Path
    glob: str


@dataclass
class RuntimeConfig:
    model: str
    temperature: float
    max_pages_per_book: int
    dpi: int
    request_timeout_seconds: int
    sleep_seconds_between_requests: float
    retry_dpi: int
    max_unusable_retries: int
    log_prompts: bool
    prompt_log_path: Path


@dataclass
class DatasetConfig:
    output_path: Path
    qas_per_page: int
    user_profile: str
    variety: dict[str, Any]
    citation: dict[str, Any]
    skipped_pages_output_path: Path


@dataclass
class PromptConfig:
    system: str


@dataclass
class AppConfig:
    input: InputConfig
    runtime: RuntimeConfig
    dataset: DatasetConfig
    prompts: PromptConfig


def load_config(config_path: Path) -> AppConfig:
    with config_path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    return AppConfig(
        input=InputConfig(
            books_dir=Path(raw["input"]["books_dir"]),
            glob=raw["input"].get("glob", "*.pdf"),
        ),
        runtime=RuntimeConfig(
            model=raw["runtime"]["model"],
            temperature=float(raw["runtime"].get("temperature", 0.2)),
            max_pages_per_book=int(raw["runtime"].get("max_pages_per_book", 10)),
            dpi=int(raw["runtime"].get("dpi", 150)),
            request_timeout_seconds=int(raw["runtime"].get("request_timeout_seconds", 120)),
            sleep_seconds_between_requests=float(
                raw["runtime"].get("sleep_seconds_between_requests", 0.4)
            ),
            retry_dpi=int(raw["runtime"].get("retry_dpi", 220)),
            max_unusable_retries=int(raw["runtime"].get("max_unusable_retries", 1)),
            log_prompts=bool(raw["runtime"].get("log_prompts", True)),
            prompt_log_path=Path(
                raw["runtime"].get("prompt_log_path", "./output/prompt_log.jsonl")
            ),
        ),
        dataset=DatasetConfig(
            output_path=Path(raw["dataset"]["output_path"]),
            qas_per_page=int(raw["dataset"].get("qas_per_page", 6)),
            user_profile=raw["dataset"].get("user_profile", ""),
            variety=raw["dataset"].get("variety", {}),
            citation=raw["dataset"].get("citation", {}),
            skipped_pages_output_path=Path(
                raw["dataset"].get("skipped_pages_output_path", "./output/skipped_pages.jsonl")
            ),
        ),
        prompts=PromptConfig(
            system=raw.get("prompts", {}).get(
                "system",
                "Create grounded QnA using the page image and return strict JSON.",
            )
        ),
    )
