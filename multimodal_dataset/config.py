"""Strongly-typed configuration loader for config.yaml."""

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
    log_api_metrics: bool
    api_metrics_log_path: Path
    generation_input_cost_per_1m_tokens_usd: float
    generation_output_cost_per_1m_tokens_usd: float
    judge_input_cost_per_1m_tokens_usd: float
    judge_output_cost_per_1m_tokens_usd: float
    checkpoint_enabled: bool
    checkpoint_path: Path
    append_mode: bool
    failed_writes_log_path: Path
    parallel_critique_workers: int
    parallel_future_timeout_seconds: int
    process_log_path: Path
    verbose_success_logs: bool


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
class QualityConfig:
    enabled: bool
    use_model_self_critique: bool
    critique_model: str
    min_grounding_score: float
    min_usefulness_score: float
    duplicate_similarity_threshold: float
    require_citation_match_if_text_available: bool
    quality_log_path: Path


@dataclass
class AnalyticsConfig:
    report_path: Path
    token_stats_path: Path


@dataclass
class AppConfig:
    input: InputConfig
    runtime: RuntimeConfig
    dataset: DatasetConfig
    prompts: PromptConfig
    quality: QualityConfig
    analytics: AnalyticsConfig


def load_config(config_path: Path) -> AppConfig:
    """Load YAML config and map it into typed dataclasses."""
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
            log_api_metrics=bool(raw["runtime"].get("log_api_metrics", True)),
            api_metrics_log_path=Path(
                raw["runtime"].get("api_metrics_log_path", "./output/api_metrics.jsonl")
            ),
            generation_input_cost_per_1m_tokens_usd=float(
                raw["runtime"].get("generation_input_cost_per_1m_tokens_usd", 5.0)
            ),
            generation_output_cost_per_1m_tokens_usd=float(
                raw["runtime"].get("generation_output_cost_per_1m_tokens_usd", 15.0)
            ),
            judge_input_cost_per_1m_tokens_usd=float(
                raw["runtime"].get("judge_input_cost_per_1m_tokens_usd", 5.0)
            ),
            judge_output_cost_per_1m_tokens_usd=float(
                raw["runtime"].get("judge_output_cost_per_1m_tokens_usd", 15.0)
            ),
            checkpoint_enabled=bool(raw["runtime"].get("checkpoint_enabled", True)),
            checkpoint_path=Path(
                raw["runtime"].get("checkpoint_path", "./output/run_checkpoint.json")
            ),
            append_mode=bool(raw["runtime"].get("append_mode", True)),
            failed_writes_log_path=Path(
                raw["runtime"].get("failed_writes_log_path", "./output/failed_writes.jsonl")
            ),
            parallel_critique_workers=int(raw["runtime"].get("parallel_critique_workers", 1)),
            parallel_future_timeout_seconds=int(
                raw["runtime"].get("parallel_future_timeout_seconds", 300)
            ),
            process_log_path=Path(
                raw["runtime"].get("process_log_path", "./output/process_log.jsonl")
            ),
            verbose_success_logs=bool(raw["runtime"].get("verbose_success_logs", True)),
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
        quality=QualityConfig(
            enabled=bool(raw.get("quality", {}).get("enabled", True)),
            use_model_self_critique=bool(
                raw.get("quality", {}).get("use_model_self_critique", True)
            ),
            critique_model=raw.get("quality", {}).get("critique_model", raw["runtime"]["model"]),
            min_grounding_score=float(raw.get("quality", {}).get("min_grounding_score", 0.55)),
            min_usefulness_score=float(raw.get("quality", {}).get("min_usefulness_score", 0.5)),
            duplicate_similarity_threshold=float(
                raw.get("quality", {}).get("duplicate_similarity_threshold", 0.9)
            ),
            require_citation_match_if_text_available=bool(
                raw.get("quality", {}).get("require_citation_match_if_text_available", True)
            ),
            quality_log_path=Path(
                raw.get("quality", {}).get("quality_log_path", "./output/quality_log.jsonl")
            ),
        ),
        analytics=AnalyticsConfig(
            report_path=Path(
                raw.get("analytics", {}).get("report_path", "./output/analytics_report.json")
            ),
            token_stats_path=Path(
                raw.get("analytics", {}).get("token_stats_path", "./output/token_stats.json")
            ),
        ),
    )
