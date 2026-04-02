"""Main orchestration module for doc-to-instruction dataset generation.

This file owns the end-to-end pipeline: page rendering, generation calls,
quality gating, telemetry, checkpoint/resume behavior, and final analytics.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI
from tqdm import tqdm

from multimodal_dataset.analytics import write_analytics_report
from multimodal_dataset.chatml import append_jsonl, qa_to_chatml_record
from multimodal_dataset.config import AppConfig, load_config
from multimodal_dataset.evidence_packs import build_evidence_packs_for_book
from multimodal_dataset.openai_client import generate_qa_batch
from multimodal_dataset.page_artifacts import build_page_artifact
from multimodal_dataset.pdf_pages import (
    extract_page_text,
    get_pdf_page_count,
    iter_pdf_pages_as_data_urls,
    render_single_page_as_data_url,
)
from multimodal_dataset.quality import (
    critique_qa_item,
    critique_cross_page_item,
    has_citation_match,
    heuristic_usefulness_score,
    normalize_text,
    text_similarity,
)
from multimodal_dataset.synthesis import synthesize_cross_page_batch


UNUSABLE_STATUSES = {"blank", "unreadable"}


def _build_run_id() -> str:
    """Create a human-readable run identifier used across artifacts."""
    return time.strftime("run_%Y%m%d_%H%M%S")


def _safe_append_jsonl(
    *,
    path: Path,
    record: dict[str, Any],
    failed_writes_path: Path,
) -> None:
    """Append a JSONL record; if write fails, log fallback failure record."""
    try:
        append_jsonl(path, record)
    except Exception as exc:  # pragma: no cover - defensive fallback
        fallback_record = {
            "timestamp_epoch": int(time.time()),
            "target_path": str(path),
            "error": repr(exc),
            "record": record,
        }
        append_jsonl(failed_writes_path, fallback_record)


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    """Atomically write a JSON file via temp file + replace."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(path.suffix + ".tmp")
    with temp_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
        f.flush()
        os.fsync(f.fileno())
    os.replace(temp_path, path)


def _log_process_event(
    *,
    process_log_path: Path,
    failed_writes_path: Path,
    run_id: str,
    event: str,
    payload: dict[str, Any],
    verbose: bool,
) -> None:
    """Log structured lifecycle/process events to process_log.jsonl."""
    entry = {
        "timestamp_epoch": int(time.time()),
        "run_id": run_id,
        "event": event,
        **payload,
    }
    _safe_append_jsonl(
        path=process_log_path,
        failed_writes_path=failed_writes_path,
        record=entry,
    )
    if verbose:
        print(f"[{event}] {json.dumps(payload, ensure_ascii=False)}")


def _load_checkpoint(path: Path) -> dict[str, Any] | None:
    """Load checkpoint state if present, otherwise return None."""
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _save_checkpoint(path: Path, state: dict[str, Any]) -> None:
    """Persist checkpoint state atomically."""
    _atomic_write_json(path, state)


def _iter_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def _cross_quote_matches(item: dict[str, Any], evidence_pack: dict[str, Any]) -> bool:
    page_to_text = {
        int(p.get("page")): str(p.get("text", "")).lower()
        for p in evidence_pack.get("page_texts", [])
    }
    quotes = item.get("evidence_quotes", [])
    if not quotes:
        return False
    for q in quotes:
        page = int(q.get("page", -1))
        quote = str(q.get("quote", "")).strip().lower()
        if not quote or page not in page_to_text:
            return False
        if quote not in page_to_text[page]:
            return False
    return True


def _build_user_prompt(cfg: AppConfig, *, book_name: str, page_number: int) -> str:
    """Build the generation user prompt from config and page context."""
    variety = json.dumps(cfg.dataset.variety, ensure_ascii=False)
    citation = json.dumps(cfg.dataset.citation, ensure_ascii=False)
    return (
        f"Generate exactly {cfg.dataset.qas_per_page} diverse QnA pairs from this single page image.\n"
        f"Book: {book_name}\n"
        f"Page: {page_number}\n"
        f"User profile preference: {cfg.dataset.user_profile}\n"
        f"Variety preferences (JSON): {variety}\n"
        f"Citation preferences (JSON): {citation}\n"
        "Rules:\n"
        "1) Only use visible page content.\n"
        "2) First classify page_status as one of: usable, blank, unreadable, index_only, image_only.\n"
        "3) If page is blank/unreadable, set items to [].\n"
        "4) Keep each answer useful for instruction tuning.\n"
        "5) Include citation_quote as a short supporting quote when possible.\n"
        "6) page_status_reason must explain why the status was chosen.\n"
    )


def get_prompt_preview(cfg: AppConfig, *, book_name: str, page_number: int) -> dict[str, str]:
    """Expose generated prompts for CLI inspection/debug."""
    return {
        "system_prompt": cfg.prompts.system,
        "user_prompt": _build_user_prompt(cfg, book_name=book_name, page_number=page_number),
    }


def _iter_books(cfg: AppConfig) -> list[Path]:
    """Resolve and validate target PDF list from config input block."""
    books_dir = cfg.input.books_dir
    if not books_dir.exists():
        raise FileNotFoundError(f"Books directory not found: {books_dir}")
    return sorted(books_dir.glob(cfg.input.glob))


def _record_skip(
    *,
    skipped_path: Path,
    failed_writes_path: Path,
    run_id: str,
    book_name: str,
    page_number: int,
    status: str,
    reason: str,
    attempt_dpi: int,
) -> None:
    """Write skipped page record when page status is unusable."""
    _safe_append_jsonl(
        path=skipped_path,
        failed_writes_path=failed_writes_path,
        record={
            "run_id": run_id,
            "source_book": book_name,
            "source_page": page_number,
            "status": status,
            "reason": reason,
            "attempt_dpi": attempt_dpi,
            "timestamp_epoch": int(time.time()),
        },
    )


def _log_prompt_request(
    *,
    prompt_log_path: Path,
    failed_writes_path: Path,
    run_id: str,
    book_name: str,
    page_number: int,
    attempt: int,
    dpi: int,
    system_prompt: str,
    user_prompt: str,
) -> None:
    """Write prompt payload for reproducibility/debugging."""
    _safe_append_jsonl(
        path=prompt_log_path,
        failed_writes_path=failed_writes_path,
        record={
            "run_id": run_id,
            "timestamp_epoch": int(time.time()),
            "source_book": book_name,
            "source_page": page_number,
            "attempt": attempt,
            "dpi": dpi,
            "system_prompt": system_prompt,
            "user_prompt": user_prompt,
        },
    )


def _log_api_metric(
    *,
    api_metrics_log_path: Path,
    failed_writes_path: Path,
    run_id: str,
    call_type: str,
    book_name: str,
    page_number: int,
    attempt: int,
    metrics: dict[str, Any],
) -> None:
    """Write one API call telemetry record (latency/tokens/cost)."""
    _safe_append_jsonl(
        path=api_metrics_log_path,
        failed_writes_path=failed_writes_path,
        record={
            "run_id": run_id,
            "timestamp_epoch": int(time.time()),
            "call_type": call_type,
            "source_book": book_name,
            "source_page": page_number,
            "attempt": attempt,
            **metrics,
        },
    )


def _generate_with_retry(
    *,
    client: OpenAI,
    cfg: AppConfig,
    failed_writes_path: Path,
    run_id: str,
    book_path: Path,
    book_name: str,
    page_number: int,
    initial_image_data_url: str,
) -> tuple[dict[str, Any], int, str, list[dict[str, Any]]]:
    """Generate QA payload with optional higher-DPI retry for unusable pages."""
    preview = get_prompt_preview(cfg, book_name=book_name, page_number=page_number)
    prompt = preview["user_prompt"]
    system_prompt = preview["system_prompt"]
    attempt = 1
    if cfg.runtime.log_prompts:
        _log_prompt_request(
            prompt_log_path=cfg.runtime.prompt_log_path,
            failed_writes_path=failed_writes_path,
            run_id=run_id,
            book_name=book_name,
            page_number=page_number,
            attempt=attempt,
            dpi=cfg.runtime.dpi,
            system_prompt=system_prompt,
            user_prompt=prompt,
        )

    payload, metrics = generate_qa_batch(
        client=client,
        model=cfg.runtime.model,
        temperature=cfg.runtime.temperature,
        system_prompt=system_prompt,
        user_prompt=prompt,
        image_data_url=initial_image_data_url,
        input_cost_per_1m_tokens_usd=cfg.runtime.generation_input_cost_per_1m_tokens_usd,
        output_cost_per_1m_tokens_usd=cfg.runtime.generation_output_cost_per_1m_tokens_usd,
    )
    call_metrics: list[dict[str, Any]] = [
        {"call_type": "generation", "attempt": attempt, **metrics}
    ]
    last_dpi = cfg.runtime.dpi
    current_image_data_url = initial_image_data_url

    retries = 0
    while (
        payload.get("page_status") in UNUSABLE_STATUSES
        and retries < cfg.runtime.max_unusable_retries
    ):
        retries += 1
        attempt += 1
        last_dpi = cfg.runtime.retry_dpi
        retry_image_data_url = render_single_page_as_data_url(
            book_path, page_number=page_number, dpi=cfg.runtime.retry_dpi
        )
        current_image_data_url = retry_image_data_url
        if cfg.runtime.log_prompts:
            _log_prompt_request(
                prompt_log_path=cfg.runtime.prompt_log_path,
                failed_writes_path=failed_writes_path,
                run_id=run_id,
                book_name=book_name,
                page_number=page_number,
                attempt=attempt,
                dpi=cfg.runtime.retry_dpi,
                system_prompt=system_prompt,
                user_prompt=prompt,
            )
        payload, metrics = generate_qa_batch(
            client=client,
            model=cfg.runtime.model,
            temperature=cfg.runtime.temperature,
            system_prompt=system_prompt,
            user_prompt=prompt,
            image_data_url=retry_image_data_url,
            input_cost_per_1m_tokens_usd=cfg.runtime.generation_input_cost_per_1m_tokens_usd,
            output_cost_per_1m_tokens_usd=cfg.runtime.generation_output_cost_per_1m_tokens_usd,
        )
        call_metrics.append({"call_type": "generation", "attempt": attempt, **metrics})

    return payload, last_dpi, current_image_data_url, call_metrics


def _log_quality_decision(
    *,
    quality_log_path: Path,
    failed_writes_path: Path,
    run_id: str,
    book_name: str,
    page_number: int,
    qa_item: dict[str, Any],
    accepted: bool,
    reasons: list[str],
    grounding_score: float,
    usefulness_score: float,
) -> None:
    """Write per-item quality gate verdict for auditability."""
    _safe_append_jsonl(
        path=quality_log_path,
        failed_writes_path=failed_writes_path,
        record={
            "run_id": run_id,
            "timestamp_epoch": int(time.time()),
            "source_book": book_name,
            "source_page": page_number,
            "accepted": accepted,
            "reasons": reasons,
            "grounding_score": grounding_score,
            "usefulness_score": usefulness_score,
            "question": qa_item.get("question", ""),
        },
    )


def _run_critique_task(
    *,
    api_key: str,
    timeout_seconds: int,
    cfg: AppConfig,
    final_image_data_url: str,
    qa_item: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Execute one judge-model critique task (thread-safe client per task)."""
    # Use a per-task client to avoid cross-thread client contention.
    thread_client = OpenAI(api_key=api_key, timeout=timeout_seconds)
    return critique_qa_item(
        client=thread_client,
        model=cfg.quality.critique_model,
        system_prompt=(
            "You are a strict dataset quality auditor. "
            "Only approve grounded and instruction-useful QA pairs."
        ),
        image_data_url=final_image_data_url,
        qa_item=qa_item,
        input_cost_per_1m_tokens_usd=cfg.runtime.judge_input_cost_per_1m_tokens_usd,
        output_cost_per_1m_tokens_usd=cfg.runtime.judge_output_cost_per_1m_tokens_usd,
    )


def run_pipeline(
    config_path: Path,
    dry_run: bool,
    resume: bool,
    skip_cross_page: bool = False,
    cross_page_only: bool = False,
) -> None:
    """Run full pipeline with logging, checkpoints, quality gate, and analytics."""
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY not found. Put it in your .env file.")

    cfg = load_config(config_path)
    books = _iter_books(cfg)
    if not books:
        raise RuntimeError(
            f"No PDFs found. Check input.books_dir='{cfg.input.books_dir}' and glob='{cfg.input.glob}'."
        )

    out_path = cfg.dataset.output_path
    skipped_path = cfg.dataset.skipped_pages_output_path
    cross_out_path = cfg.cross_page.output_path
    page_artifact_path = cfg.cross_page.artifact_path
    cross_quality_log_path = cfg.cross_page.quality_log_path
    prompt_log_path = cfg.runtime.prompt_log_path
    api_metrics_log_path = cfg.runtime.api_metrics_log_path
    process_log_path = cfg.runtime.process_log_path
    quality_log_path = cfg.quality.quality_log_path
    analytics_path = cfg.analytics.report_path
    token_stats_path = cfg.analytics.token_stats_path
    checkpoint_path = cfg.runtime.checkpoint_path
    failed_writes_path = cfg.runtime.failed_writes_log_path

    prior_checkpoint = (
        _load_checkpoint(checkpoint_path)
        if (resume and cfg.runtime.checkpoint_enabled and checkpoint_path.exists())
        else None
    )
    run_id = str(prior_checkpoint["run_id"]) if prior_checkpoint else _build_run_id()

    if not cfg.runtime.append_mode:
        if out_path.exists():
            out_path.unlink()
        if skipped_path.exists():
            skipped_path.unlink()
        if cfg.runtime.log_prompts and prompt_log_path.exists():
            prompt_log_path.unlink()
        if cfg.runtime.log_api_metrics and api_metrics_log_path.exists():
            api_metrics_log_path.unlink()
        if process_log_path.exists():
            process_log_path.unlink()
        if cfg.quality.enabled and quality_log_path.exists():
            quality_log_path.unlink()
        if cfg.cross_page.enabled and cross_out_path.exists():
            cross_out_path.unlink()
        if cfg.cross_page.enabled and page_artifact_path.exists():
            page_artifact_path.unlink()
        if cfg.cross_page.enabled and cross_quality_log_path.exists():
            cross_quality_log_path.unlink()
        if failed_writes_path.exists():
            failed_writes_path.unlink()

    client = OpenAI(api_key=api_key, timeout=cfg.runtime.request_timeout_seconds)
    counters = dict(prior_checkpoint.get("counters", {})) if prior_checkpoint else {}
    written_records = int(counters.get("written_records", 0))
    local_records_accepted = int(counters.get("local_records_accepted", written_records))
    skipped_records = int(counters.get("skipped_records", 0))
    total_pages_seen = int(counters.get("total_pages_seen", 0))
    generated_candidates = int(counters.get("generated_candidates", 0))
    duplicates_filtered = int(counters.get("duplicates_filtered", 0))
    quality_rejected = int(counters.get("quality_rejected", 0))
    cross_page_candidates = int(counters.get("cross_page_candidates", 0))
    cross_page_accepted = int(counters.get("cross_page_accepted", 0))
    cross_page_duplicates_filtered = int(counters.get("cross_page_duplicates_filtered", 0))
    cross_page_quality_rejected = int(counters.get("cross_page_quality_rejected", 0))
    total_estimated_cost_usd = float(counters.get("total_estimated_cost_usd", 0.0))
    total_api_calls = int(counters.get("total_api_calls", 0))
    total_generation_calls = int(counters.get("total_generation_calls", 0))
    total_critique_calls = int(counters.get("total_critique_calls", 0))
    total_api_latency_ms = int(counters.get("total_api_latency_ms", 0))
    total_input_tokens = int(counters.get("total_input_tokens", 0))
    total_output_tokens = int(counters.get("total_output_tokens", 0))
    generation_input_tokens = int(counters.get("generation_input_tokens", 0))
    generation_output_tokens = int(counters.get("generation_output_tokens", 0))
    critique_input_tokens = int(counters.get("critique_input_tokens", 0))
    critique_output_tokens = int(counters.get("critique_output_tokens", 0))

    question_type_counts: Counter[str] = Counter(
        prior_checkpoint.get("question_type_counts", {}) if prior_checkpoint else {}
    )
    difficulty_counts: Counter[str] = Counter(
        prior_checkpoint.get("difficulty_counts", {}) if prior_checkpoint else {}
    )
    answer_lengths: list[int] = (
        list(prior_checkpoint.get("answer_lengths", [])) if prior_checkpoint else []
    )
    per_book_yield: defaultdict[str, int] = defaultdict(
        int, prior_checkpoint.get("per_book_yield", {}) if prior_checkpoint else {}
    )
    seen_pairs: list[str] = list(prior_checkpoint.get("seen_pairs", [])) if prior_checkpoint else []
    processed_pages_raw: dict[str, list[int]] = (
        prior_checkpoint.get("processed_pages", {}) if prior_checkpoint else {}
    )
    processed_pages: defaultdict[str, set[int]] = defaultdict(
        set,
        {book_name: set(pages) for book_name, pages in processed_pages_raw.items()},
    )
    cross_page_completed_books = set(
        prior_checkpoint.get("cross_page_completed_books", []) if prior_checkpoint else []
    )
    processed_pack_ids_raw: dict[str, list[str]] = (
        prior_checkpoint.get("processed_pack_ids", {}) if prior_checkpoint else {}
    )
    processed_pack_ids: defaultdict[str, set[str]] = defaultdict(
        set,
        {book_name: set(ids) for book_name, ids in processed_pack_ids_raw.items()},
    )
    cross_seen_signatures: list[str] = list(
        prior_checkpoint.get("cross_seen_signatures", []) if prior_checkpoint else []
    )
    local_rejection_reasons: Counter[str] = Counter(
        prior_checkpoint.get("local_rejection_reasons", {}) if prior_checkpoint else {}
    )
    cross_rejection_reasons: Counter[str] = Counter(
        prior_checkpoint.get("cross_rejection_reasons", {}) if prior_checkpoint else {}
    )
    synthesis_type_counts: Counter[str] = Counter(
        prior_checkpoint.get("synthesis_type_counts", {}) if prior_checkpoint else {}
    )
    evidence_pack_page_counts: list[int] = list(
        prior_checkpoint.get("evidence_pack_page_counts", []) if prior_checkpoint else []
    )
    book_page_targets: dict[str, int] = {}
    for book in books:
        page_count = get_pdf_page_count(book)
        target_pages = min(page_count, 1 if dry_run else cfg.runtime.max_pages_per_book)
        book_page_targets[book.name] = target_pages

    remaining_pages_estimate = sum(
        max(0, target - len(processed_pages.get(book_name, set())))
        for book_name, target in book_page_targets.items()
    )
    estimated_candidate_samples = remaining_pages_estimate * cfg.dataset.qas_per_page
    historical_acceptance_rate = (
        written_records / generated_candidates if generated_candidates > 0 else 0.0
    )
    projected_final_samples = int(
        estimated_candidate_samples
        * (historical_acceptance_rate if historical_acceptance_rate > 0 else 0.4)
    )

    _log_process_event(
        process_log_path=process_log_path,
        failed_writes_path=failed_writes_path,
        run_id=run_id,
        event="run_started",
        verbose=cfg.runtime.verbose_success_logs,
        payload={
            "books": [book.name for book in books],
            "book_page_targets": book_page_targets,
            "remaining_pages_estimate": remaining_pages_estimate,
            "estimated_candidate_samples": estimated_candidate_samples,
            "projected_final_samples": projected_final_samples,
            "resume": resume,
        },
    )

    def save_checkpoint_state(status: str) -> None:
        if not cfg.runtime.checkpoint_enabled:
            return
        state = {
            "run_id": run_id,
            "status": status,
            "updated_at_epoch": int(time.time()),
            "processed_pages": {
                book_name: sorted(list(page_set))
                for book_name, page_set in processed_pages.items()
            },
            "cross_page_completed_books": sorted(list(cross_page_completed_books)),
            "processed_pack_ids": {
                book_name: sorted(list(pack_ids))
                for book_name, pack_ids in processed_pack_ids.items()
            },
            "cross_seen_signatures": cross_seen_signatures,
            "question_type_counts": dict(question_type_counts),
            "difficulty_counts": dict(difficulty_counts),
            "local_rejection_reasons": dict(local_rejection_reasons),
            "cross_rejection_reasons": dict(cross_rejection_reasons),
            "synthesis_type_counts": dict(synthesis_type_counts),
            "evidence_pack_page_counts": evidence_pack_page_counts,
            "answer_lengths": answer_lengths,
            "per_book_yield": dict(per_book_yield),
            "seen_pairs": seen_pairs,
            "counters": {
                "written_records": written_records,
                "local_records_accepted": local_records_accepted,
                "skipped_records": skipped_records,
                "total_pages_seen": total_pages_seen,
                "generated_candidates": generated_candidates,
                "duplicates_filtered": duplicates_filtered,
                "quality_rejected": quality_rejected,
                "cross_page_candidates": cross_page_candidates,
                "cross_page_accepted": cross_page_accepted,
                "cross_page_duplicates_filtered": cross_page_duplicates_filtered,
                "cross_page_quality_rejected": cross_page_quality_rejected,
                "total_estimated_cost_usd": total_estimated_cost_usd,
                "total_api_calls": total_api_calls,
                "total_generation_calls": total_generation_calls,
                "total_critique_calls": total_critique_calls,
                "total_api_latency_ms": total_api_latency_ms,
                "total_input_tokens": total_input_tokens,
                "total_output_tokens": total_output_tokens,
                "generation_input_tokens": generation_input_tokens,
                "generation_output_tokens": generation_output_tokens,
                "critique_input_tokens": critique_input_tokens,
                "critique_output_tokens": critique_output_tokens,
            },
        }
        _save_checkpoint(checkpoint_path, state)

    save_checkpoint_state("running")

    if not cross_page_only:
        for book in books:
            page_iter = iter_pdf_pages_as_data_urls(
                book,
                dpi=cfg.runtime.dpi,
                max_pages=1 if dry_run else cfg.runtime.max_pages_per_book,
            )
            for page in tqdm(page_iter, desc=f"Processing {book.name}"):
                if page.page_number in processed_pages[book.name]:
                    continue
                total_pages_seen += 1
                payload, used_dpi, final_image_data_url, generation_metrics = _generate_with_retry(
                    client=client,
                    cfg=cfg,
                    failed_writes_path=failed_writes_path,
                    run_id=run_id,
                    book_path=book,
                    book_name=book.name,
                    page_number=page.page_number,
                    initial_image_data_url=page.image_data_url,
                )
                for gm in generation_metrics:
                    total_api_calls += 1
                    total_generation_calls += 1
                    total_api_latency_ms += int(gm.get("latency_ms", 0))
                    total_input_tokens += int(gm.get("input_tokens", 0))
                    total_output_tokens += int(gm.get("output_tokens", 0))
                    generation_input_tokens += int(gm.get("input_tokens", 0))
                    generation_output_tokens += int(gm.get("output_tokens", 0))
                    total_estimated_cost_usd += float(gm.get("estimated_cost_usd", 0.0))
                    if cfg.runtime.log_api_metrics:
                        _log_api_metric(
                            api_metrics_log_path=api_metrics_log_path,
                            failed_writes_path=failed_writes_path,
                            run_id=run_id,
                            call_type=str(gm.get("call_type", "generation")),
                            book_name=book.name,
                            page_number=page.page_number,
                            attempt=int(gm.get("attempt", 1)),
                            metrics=gm,
                        )
                    _log_process_event(
                        process_log_path=process_log_path,
                        failed_writes_path=failed_writes_path,
                        run_id=run_id,
                        event="generation_call_success",
                        verbose=cfg.runtime.verbose_success_logs,
                        payload={
                            "book": book.name,
                            "page": page.page_number,
                            "attempt": int(gm.get("attempt", 1)),
                            "latency_ms": int(gm.get("latency_ms", 0)),
                            "input_tokens": int(gm.get("input_tokens", 0)),
                            "output_tokens": int(gm.get("output_tokens", 0)),
                            "estimated_cost_usd": float(gm.get("estimated_cost_usd", 0.0)),
                        },
                    )

                status = payload.get("page_status", "unreadable")
                reason = payload.get("page_status_reason", "")
                items = payload.get("items", [])
                generated_candidates += len(items)

                if status in UNUSABLE_STATUSES:
                    _record_skip(
                        skipped_path=skipped_path,
                        failed_writes_path=failed_writes_path,
                        run_id=run_id,
                        book_name=book.name,
                        page_number=page.page_number,
                        status=status,
                        reason=reason,
                        attempt_dpi=used_dpi,
                    )
                    skipped_records += 1
                    if cfg.cross_page.enabled:
                        page_artifact = build_page_artifact(
                            run_id=run_id,
                            source_book=book.name,
                            source_page=page.page_number,
                            page_status=status,
                            page_status_reason=reason,
                            page_text=extract_page_text(book, page_number=page.page_number),
                            accepted_qas=[],
                        )
                        _safe_append_jsonl(
                            path=page_artifact_path,
                            failed_writes_path=failed_writes_path,
                            record=page_artifact,
                        )
                    _log_process_event(
                        process_log_path=process_log_path,
                        failed_writes_path=failed_writes_path,
                        run_id=run_id,
                        event="page_skipped",
                        verbose=cfg.runtime.verbose_success_logs,
                        payload={
                            "book": book.name,
                            "page": page.page_number,
                            "status": status,
                            "reason": reason,
                        },
                    )
                    processed_pages[book.name].add(page.page_number)
                    save_checkpoint_state("running")
                    time.sleep(cfg.runtime.sleep_seconds_between_requests)
                    continue

                page_text = extract_page_text(book, page_number=page.page_number)
                local_accepted_qas: list[dict[str, Any]] = []
                pending_items: list[dict[str, Any]] = []
                for qa in items:
                    question = qa.get("question", "").strip()
                    answer = qa.get("answer", "").strip()
                    citation_quote = qa.get("citation_quote", "").strip()
                    signature = normalize_text(f"{question} || {answer}")

                    duplicate_hit = False
                    if signature in seen_pairs:
                        duplicate_hit = True
                    else:
                        for prev in seen_pairs:
                            if (
                                text_similarity(signature, prev)
                                >= cfg.quality.duplicate_similarity_threshold
                            ):
                                duplicate_hit = True
                                break
                    if duplicate_hit:
                        duplicates_filtered += 1
                        local_rejection_reasons["duplicate_or_near_duplicate"] += 1
                        _log_process_event(
                            process_log_path=process_log_path,
                            failed_writes_path=failed_writes_path,
                            run_id=run_id,
                            event="record_rejected_duplicate",
                            verbose=cfg.runtime.verbose_success_logs,
                            payload={
                                "book": book.name,
                                "page": page.page_number,
                                "question": question[:120],
                            },
                        )
                        if cfg.quality.enabled:
                            _log_quality_decision(
                                quality_log_path=quality_log_path,
                                failed_writes_path=failed_writes_path,
                                run_id=run_id,
                                book_name=book.name,
                                page_number=page.page_number,
                                qa_item=qa,
                                accepted=False,
                                reasons=["duplicate_or_near_duplicate"],
                                grounding_score=0.0,
                                usefulness_score=0.0,
                            )
                        continue

                    pending_items.append(
                        {
                            "qa": qa,
                            "question": question,
                            "answer": answer,
                            "citation_quote": citation_quote,
                            "signature": signature,
                        }
                    )

                critique_results: dict[int, dict[str, Any]] = {}
                if (
                    cfg.quality.enabled
                    and cfg.quality.use_model_self_critique
                    and pending_items
                ):
                    worker_count = max(1, cfg.runtime.parallel_critique_workers)
                    if worker_count == 1 or len(pending_items) == 1:
                        for idx, pending in enumerate(pending_items):
                            try:
                                critique, critique_metrics = _run_critique_task(
                                    api_key=api_key,
                                    timeout_seconds=cfg.runtime.request_timeout_seconds,
                                    cfg=cfg,
                                    final_image_data_url=final_image_data_url,
                                    qa_item=pending["qa"],
                                )
                                critique_results[idx] = {
                                    "critique": critique,
                                    "metrics": critique_metrics,
                                    "error": None,
                                }
                            except Exception as exc:
                                critique_results[idx] = {
                                    "critique": {},
                                    "metrics": {},
                                    "error": repr(exc),
                                }
                    else:
                        with concurrent.futures.ThreadPoolExecutor(
                            max_workers=min(worker_count, len(pending_items))
                        ) as executor:
                            future_to_idx = {
                                executor.submit(
                                    _run_critique_task,
                                    api_key=api_key,
                                    timeout_seconds=cfg.runtime.request_timeout_seconds,
                                    cfg=cfg,
                                    final_image_data_url=final_image_data_url,
                                    qa_item=pending["qa"],
                                ): idx
                                for idx, pending in enumerate(pending_items)
                            }
                            try:
                                for future in concurrent.futures.as_completed(
                                    future_to_idx,
                                    timeout=cfg.runtime.parallel_future_timeout_seconds,
                                ):
                                    idx = future_to_idx[future]
                                    try:
                                        critique, critique_metrics = future.result()
                                        critique_results[idx] = {
                                            "critique": critique,
                                            "metrics": critique_metrics,
                                            "error": None,
                                        }
                                    except Exception as exc:
                                        critique_results[idx] = {
                                            "critique": {},
                                            "metrics": {},
                                            "error": repr(exc),
                                        }
                            except concurrent.futures.TimeoutError:
                                pass
                            for future, idx in future_to_idx.items():
                                if idx in critique_results:
                                    continue
                                future.cancel()
                                critique_results[idx] = {
                                    "critique": {},
                                    "metrics": {},
                                    "error": "critique_future_timeout_or_cancelled",
                                }

                for idx, pending in enumerate(pending_items):
                    qa = pending["qa"]
                    question = pending["question"]
                    answer = pending["answer"]
                    citation_quote = pending["citation_quote"]
                    signature = pending["signature"]

                    reasons: list[str] = []
                    grounding_score = 1.0
                    usefulness_score = heuristic_usefulness_score(question, answer)

                    if cfg.quality.enabled and cfg.quality.use_model_self_critique:
                        critique_entry = critique_results.get(idx, {})
                        critique_error = critique_entry.get("error")
                        if critique_error:
                            reasons.append("critique_call_failed")
                            grounding_score = 0.0
                            usefulness_score = 0.0
                        else:
                            critique = critique_entry.get("critique", {})
                            critique_metrics = critique_entry.get("metrics", {})
                            total_api_calls += 1
                            total_critique_calls += 1
                            total_api_latency_ms += int(critique_metrics.get("latency_ms", 0))
                            total_input_tokens += int(critique_metrics.get("input_tokens", 0))
                            total_output_tokens += int(critique_metrics.get("output_tokens", 0))
                            critique_input_tokens += int(critique_metrics.get("input_tokens", 0))
                            critique_output_tokens += int(critique_metrics.get("output_tokens", 0))
                            total_estimated_cost_usd += float(
                                critique_metrics.get("estimated_cost_usd", 0.0)
                            )
                            if cfg.runtime.log_api_metrics:
                                _log_api_metric(
                                    api_metrics_log_path=api_metrics_log_path,
                                    failed_writes_path=failed_writes_path,
                                    run_id=run_id,
                                    call_type="critique",
                                    book_name=book.name,
                                    page_number=page.page_number,
                                    attempt=1,
                                    metrics=critique_metrics,
                                )
                            _log_process_event(
                                process_log_path=process_log_path,
                                failed_writes_path=failed_writes_path,
                                run_id=run_id,
                                event="critique_call_success",
                                verbose=cfg.runtime.verbose_success_logs,
                                payload={
                                    "book": book.name,
                                    "page": page.page_number,
                                    "latency_ms": int(critique_metrics.get("latency_ms", 0)),
                                    "input_tokens": int(critique_metrics.get("input_tokens", 0)),
                                    "output_tokens": int(critique_metrics.get("output_tokens", 0)),
                                    "estimated_cost_usd": float(
                                        critique_metrics.get("estimated_cost_usd", 0.0)
                                    ),
                                },
                            )
                            grounding_score = float(critique.get("grounding_score", grounding_score))
                            usefulness_score = float(
                                critique.get("usefulness_score", usefulness_score)
                            )
                            if not bool(critique.get("grounded", True)):
                                reasons.append("model_marked_not_grounded")
                            if not bool(critique.get("useful", True)):
                                reasons.append("model_marked_not_useful")

                    text_available = len(page_text.strip()) > 20
                    if (
                        cfg.quality.enabled
                        and cfg.quality.require_citation_match_if_text_available
                        and text_available
                        and not has_citation_match(page_text, citation_quote)
                    ):
                        grounding_score = min(grounding_score, 0.2)
                        reasons.append("citation_not_found_in_page_text")

                    if grounding_score < cfg.quality.min_grounding_score:
                        reasons.append("grounding_score_below_threshold")
                    if usefulness_score < cfg.quality.min_usefulness_score:
                        reasons.append("usefulness_score_below_threshold")

                    accepted = len(reasons) == 0
                    if cfg.quality.enabled:
                        _log_quality_decision(
                            quality_log_path=quality_log_path,
                            failed_writes_path=failed_writes_path,
                            run_id=run_id,
                            book_name=book.name,
                            page_number=page.page_number,
                            qa_item=qa,
                            accepted=accepted,
                            reasons=reasons,
                            grounding_score=grounding_score,
                            usefulness_score=usefulness_score,
                        )
                    if not accepted:
                        quality_rejected += 1
                        if reasons:
                            for reason_value in reasons:
                                local_rejection_reasons[reason_value] += 1
                        else:
                            local_rejection_reasons["rejected_without_reason"] += 1
                        _log_process_event(
                            process_log_path=process_log_path,
                            failed_writes_path=failed_writes_path,
                            run_id=run_id,
                            event="record_rejected_quality",
                            verbose=cfg.runtime.verbose_success_logs,
                            payload={
                                "book": book.name,
                                "page": page.page_number,
                                "reasons": reasons,
                                "grounding_score": grounding_score,
                                "usefulness_score": usefulness_score,
                            },
                        )
                        continue

                    record = qa_to_chatml_record(
                        qa_item=qa,
                        book_name=book.name,
                        page_number=page.page_number,
                        user_profile=cfg.dataset.user_profile,
                    )
                    record.setdefault("metadata", {})
                    record["metadata"]["run_id"] = run_id
                    _safe_append_jsonl(
                        path=out_path,
                        failed_writes_path=failed_writes_path,
                        record=record,
                    )
                    _log_process_event(
                        process_log_path=process_log_path,
                        failed_writes_path=failed_writes_path,
                        run_id=run_id,
                        event="record_accepted",
                        verbose=cfg.runtime.verbose_success_logs,
                        payload={
                            "book": book.name,
                            "page": page.page_number,
                            "question_type": qa.get("question_type", "unknown"),
                            "difficulty": qa.get("difficulty", "unknown"),
                        },
                    )
                    written_records += 1
                    local_records_accepted += 1
                    seen_pairs.append(signature)
                    question_type_counts[qa.get("question_type", "unknown")] += 1
                    difficulty_counts[qa.get("difficulty", "unknown")] += 1
                    answer_lengths.append(len(answer.split()))
                    per_book_yield[book.name] += 1
                    local_accepted_qas.append(qa)

                if cfg.cross_page.enabled:
                    page_artifact = build_page_artifact(
                        run_id=run_id,
                        source_book=book.name,
                        source_page=page.page_number,
                        page_status=status,
                        page_status_reason=reason,
                        page_text=page_text,
                        accepted_qas=local_accepted_qas,
                    )
                    _safe_append_jsonl(
                        path=page_artifact_path,
                        failed_writes_path=failed_writes_path,
                        record=page_artifact,
                    )

                processed_pages[book.name].add(page.page_number)
                _log_process_event(
                    process_log_path=process_log_path,
                    failed_writes_path=failed_writes_path,
                    run_id=run_id,
                    event="page_processed",
                    verbose=cfg.runtime.verbose_success_logs,
                    payload={
                        "book": book.name,
                        "page": page.page_number,
                        "records_so_far": written_records,
                        "skipped_pages_so_far": skipped_records,
                        "quality_rejected_so_far": quality_rejected,
                    },
                )
                save_checkpoint_state("running")
                time.sleep(cfg.runtime.sleep_seconds_between_requests)

    # Stage 2: cross-page synthesis over deterministic evidence packs.
    if cfg.cross_page.enabled and not skip_cross_page:
        artifacts_all = _iter_jsonl(page_artifact_path)
        artifacts_by_book: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
        for artifact in artifacts_all:
            if str(artifact.get("run_id")) != run_id:
                continue
            artifacts_by_book[str(artifact.get("source_book", ""))].append(artifact)

        for book in books:
            if book.name in cross_page_completed_books:
                continue
            book_artifacts = artifacts_by_book.get(book.name, [])
            packs = build_evidence_packs_for_book(
                source_book=book.name,
                page_artifacts=book_artifacts,
                min_pages_per_pack=cfg.cross_page.min_pages_per_pack,
                max_pages_per_pack=cfg.cross_page.max_pages_per_pack,
                pack_overlap_window=cfg.cross_page.pack_overlap_window,
            )
            _log_process_event(
                process_log_path=process_log_path,
                failed_writes_path=failed_writes_path,
                run_id=run_id,
                event="cross_page_packs_built",
                verbose=cfg.runtime.verbose_success_logs,
                payload={"book": book.name, "pack_count": len(packs)},
            )

            for pack in packs:
                pack_id = str(pack.get("pack_id"))
                if pack_id in processed_pack_ids[book.name]:
                    continue
                evidence_pack_page_counts.append(len(pack.get("source_pages", [])))

                synth_payload, synth_metrics = synthesize_cross_page_batch(
                    client=client,
                    model=cfg.cross_page.synthesis_model,
                    temperature=cfg.cross_page.synthesis_temperature,
                    evidence_pack=pack,
                    max_cross_page_qas_per_pack=cfg.cross_page.max_cross_page_qas_per_pack,
                    max_evidence_quotes_per_item=cfg.cross_page.max_evidence_quotes_per_item,
                    use_local_qas_as_hints=cfg.cross_page.use_local_qas_as_hints,
                    input_cost_per_1m_tokens_usd=cfg.runtime.generation_input_cost_per_1m_tokens_usd,
                    output_cost_per_1m_tokens_usd=cfg.runtime.generation_output_cost_per_1m_tokens_usd,
                )
                total_api_calls += 1
                total_generation_calls += 1
                total_api_latency_ms += int(synth_metrics.get("latency_ms", 0))
                total_input_tokens += int(synth_metrics.get("input_tokens", 0))
                total_output_tokens += int(synth_metrics.get("output_tokens", 0))
                generation_input_tokens += int(synth_metrics.get("input_tokens", 0))
                generation_output_tokens += int(synth_metrics.get("output_tokens", 0))
                total_estimated_cost_usd += float(synth_metrics.get("estimated_cost_usd", 0.0))
                if cfg.runtime.log_api_metrics:
                    _log_api_metric(
                        api_metrics_log_path=api_metrics_log_path,
                        failed_writes_path=failed_writes_path,
                        run_id=run_id,
                        call_type="synthesis",
                        book_name=book.name,
                        page_number=int(pack.get("source_pages", [0])[0]),
                        attempt=1,
                        metrics=synth_metrics,
                    )

                pack_status = str(synth_payload.get("pack_status", "insufficient_evidence"))
                pack_reason = str(synth_payload.get("pack_status_reason", ""))
                if pack_status != "usable":
                    cross_rejection_reasons[f"pack_status_{pack_status}"] += 1
                    _safe_append_jsonl(
                        path=cross_quality_log_path,
                        failed_writes_path=failed_writes_path,
                        record={
                            "run_id": run_id,
                            "source_book": book.name,
                            "pack_id": pack_id,
                            "accepted": False,
                            "reasons": [pack_reason or f"pack_status_{pack_status}"],
                        },
                    )
                    processed_pack_ids[book.name].add(pack_id)
                    save_checkpoint_state("running")
                    continue

                items = list(synth_payload.get("items", []))
                cross_page_candidates += len(items)

                for item in items:
                    question = str(item.get("question", "")).strip()
                    answer = str(item.get("answer", "")).strip()
                    signature = normalize_text(f"{question} || {answer}")
                    duplicate_hit = False
                    if signature in cross_seen_signatures or signature in seen_pairs:
                        duplicate_hit = True
                    else:
                        for prev in cross_seen_signatures + seen_pairs:
                            if text_similarity(signature, prev) >= cfg.quality.duplicate_similarity_threshold:
                                duplicate_hit = True
                                break
                    if duplicate_hit:
                        cross_page_duplicates_filtered += 1
                        cross_rejection_reasons["duplicate_or_near_duplicate"] += 1
                        continue

                    reasons: list[str] = []
                    critique, critique_metrics = critique_cross_page_item(
                        client=client,
                        model=cfg.cross_page.synthesis_model,
                        evidence_pack=pack,
                        item=item,
                        input_cost_per_1m_tokens_usd=cfg.runtime.judge_input_cost_per_1m_tokens_usd,
                        output_cost_per_1m_tokens_usd=cfg.runtime.judge_output_cost_per_1m_tokens_usd,
                    )
                    total_api_calls += 1
                    total_critique_calls += 1
                    total_api_latency_ms += int(critique_metrics.get("latency_ms", 0))
                    total_input_tokens += int(critique_metrics.get("input_tokens", 0))
                    total_output_tokens += int(critique_metrics.get("output_tokens", 0))
                    critique_input_tokens += int(critique_metrics.get("input_tokens", 0))
                    critique_output_tokens += int(critique_metrics.get("output_tokens", 0))
                    total_estimated_cost_usd += float(critique_metrics.get("estimated_cost_usd", 0.0))
                    if cfg.runtime.log_api_metrics:
                        _log_api_metric(
                            api_metrics_log_path=api_metrics_log_path,
                            failed_writes_path=failed_writes_path,
                            run_id=run_id,
                            call_type="cross_page_critique",
                            book_name=book.name,
                            page_number=int(pack.get("source_pages", [0])[0]),
                            attempt=1,
                            metrics=critique_metrics,
                        )

                    grounding_score = float(critique.get("grounding_score", 0.0))
                    usefulness_score = float(critique.get("usefulness_score", 0.0))
                    multi_page_score = float(critique.get("multi_page_score", 0.0))
                    if not bool(critique.get("grounded", True)):
                        reasons.append("model_marked_not_grounded")
                    if not bool(critique.get("useful", True)):
                        reasons.append("model_marked_not_useful")
                    if not bool(critique.get("truly_multi_page", True)):
                        reasons.append("model_marked_not_truly_multi_page")

                    if cfg.cross_page.require_quote_match_if_text_available and not _cross_quote_matches(
                        item, pack
                    ):
                        reasons.append("evidence_quote_not_found_in_pack_text")
                    if grounding_score < cfg.cross_page.min_cross_page_grounding_score:
                        reasons.append("cross_page_grounding_below_threshold")
                    if usefulness_score < cfg.cross_page.min_cross_page_usefulness_score:
                        reasons.append("cross_page_usefulness_below_threshold")
                    if multi_page_score < cfg.cross_page.min_multi_page_score:
                        reasons.append("multi_page_score_below_threshold")

                    accepted = len(reasons) == 0
                    _safe_append_jsonl(
                        path=cross_quality_log_path,
                        failed_writes_path=failed_writes_path,
                        record={
                            "run_id": run_id,
                            "source_book": book.name,
                            "pack_id": pack_id,
                            "question": question,
                            "accepted": accepted,
                            "reasons": reasons,
                            "grounding_score": grounding_score,
                            "usefulness_score": usefulness_score,
                            "multi_page_score": multi_page_score,
                        },
                    )
                    if not accepted:
                        cross_page_quality_rejected += 1
                        for r in reasons:
                            cross_rejection_reasons[r] += 1
                        continue

                    record = qa_to_chatml_record(
                        qa_item=item,
                        book_name=book.name,
                        page_number=int(item.get("source_pages", [pack.get("source_pages", [1])[0]])[0]),
                        user_profile=cfg.dataset.user_profile,
                        source_pages=list(item.get("source_pages", [])),
                        evidence_quotes=list(item.get("evidence_quotes", [])),
                        synthesis_type=str(item.get("synthesis_type", "")),
                        record_level="cross_page",
                        pack_id=pack_id,
                    )
                    record.setdefault("metadata", {})
                    record["metadata"]["run_id"] = run_id
                    _safe_append_jsonl(
                        path=cross_out_path,
                        failed_writes_path=failed_writes_path,
                        record=record,
                    )
                    if cfg.cross_page.merge_into_final_dataset:
                        _safe_append_jsonl(
                            path=out_path,
                            failed_writes_path=failed_writes_path,
                            record=record,
                        )
                    cross_page_accepted += 1
                    cross_seen_signatures.append(signature)
                    synthesis_type_counts[str(item.get("synthesis_type", "unknown"))] += 1

                processed_pack_ids[book.name].add(pack_id)
                save_checkpoint_state("running")

            cross_page_completed_books.add(book.name)
            save_checkpoint_state("running")

    average_answer_length = (
        sum(answer_lengths) / len(answer_lengths) if answer_lengths else 0.0
    )
    duplicate_rate = (
        duplicates_filtered / generated_candidates if generated_candidates else 0.0
    )
    unusable_page_rate = skipped_records / total_pages_seen if total_pages_seen else 0.0
    average_api_latency_ms = total_api_latency_ms / total_api_calls if total_api_calls else 0.0
    average_input_tokens_per_call = total_input_tokens / total_api_calls if total_api_calls else 0.0
    average_output_tokens_per_call = (
        total_output_tokens / total_api_calls if total_api_calls else 0.0
    )
    average_pages_per_pack = (
        sum(evidence_pack_page_counts) / len(evidence_pack_page_counts)
        if evidence_pack_page_counts
        else 0.0
    )
    cross_duplicate_rate = (
        cross_page_duplicates_filtered / cross_page_candidates if cross_page_candidates else 0.0
    )
    analytics_report = {
        "run_id": run_id,
        "total_records": local_records_accepted + cross_page_accepted,
        "total_candidate_qas": generated_candidates,
        "local_candidates": generated_candidates,
        "local_accepted": local_records_accepted,
        "cross_page_candidates": cross_page_candidates,
        "cross_page_accepted": cross_page_accepted,
        "question_type_distribution": dict(question_type_counts),
        "difficulty_distribution": dict(difficulty_counts),
        "average_answer_length_words": round(average_answer_length, 2),
        "duplicate_rate": round(duplicate_rate, 4),
        "unusable_page_rate": round(unusable_page_rate, 4),
        "per_book_yield": dict(per_book_yield),
        "quality_rejected": quality_rejected,
        "duplicates_filtered": duplicates_filtered,
        "cross_page_duplicates_filtered": cross_page_duplicates_filtered,
        "skipped_pages": skipped_records,
        "total_pages_seen": total_pages_seen,
        "api_calls_total": total_api_calls,
        "api_calls_generation": total_generation_calls,
        "api_calls_critique": total_critique_calls,
        "api_average_latency_ms": round(average_api_latency_ms, 2),
        "api_total_input_tokens": total_input_tokens,
        "api_total_output_tokens": total_output_tokens,
        "api_average_input_tokens_per_call": round(average_input_tokens_per_call, 2),
        "api_average_output_tokens_per_call": round(average_output_tokens_per_call, 2),
        "estimated_total_cost_usd": round(total_estimated_cost_usd, 6),
        "local_rejection_reasons": dict(local_rejection_reasons),
        "cross_rejection_reasons": dict(cross_rejection_reasons),
        "average_pages_per_evidence_pack": round(average_pages_per_pack, 2),
        "synthesis_type_distribution": dict(synthesis_type_counts),
        "cross_page_duplicate_rate": round(cross_duplicate_rate, 4),
    }
    write_analytics_report(analytics_path, analytics_report)
    token_stats_report = {
        "run_id": run_id,
        "totals": {
            "input_tokens": total_input_tokens,
            "output_tokens": total_output_tokens,
            "calls": total_api_calls,
        },
        "generation": {
            "input_tokens": generation_input_tokens,
            "output_tokens": generation_output_tokens,
            "calls": total_generation_calls,
        },
        "critique": {
            "input_tokens": critique_input_tokens,
            "output_tokens": critique_output_tokens,
            "calls": total_critique_calls,
        },
    }
    _atomic_write_json(token_stats_path, token_stats_report)
    save_checkpoint_state("completed")
    _log_process_event(
        process_log_path=process_log_path,
        failed_writes_path=failed_writes_path,
        run_id=run_id,
        event="run_completed",
        verbose=cfg.runtime.verbose_success_logs,
        payload={
            "total_records": written_records,
            "total_candidate_qas": generated_candidates,
            "quality_rejected": quality_rejected,
            "duplicates_filtered": duplicates_filtered,
            "skipped_pages": skipped_records,
            "estimated_total_cost_usd": round(total_estimated_cost_usd, 6),
            "average_api_latency_ms": round(average_api_latency_ms, 2),
        },
    )

    print(f"Wrote dataset to: {out_path}")
    print(f"Wrote skipped-page log to: {skipped_path}")
    if cfg.cross_page.enabled:
        print(f"Wrote page artifacts to: {page_artifact_path}")
        print(f"Wrote cross-page dataset to: {cross_out_path}")
        print(f"Wrote cross-page quality log to: {cross_quality_log_path}")
    if cfg.runtime.log_prompts:
        print(f"Wrote prompt log to: {prompt_log_path}")
    if cfg.runtime.log_api_metrics:
        print(f"Wrote API metrics log to: {api_metrics_log_path}")
    print(f"Wrote process log to: {process_log_path}")
    print(f"Wrote token stats to: {token_stats_path}")
    if cfg.quality.enabled:
        print(f"Wrote quality log to: {quality_log_path}")
    print(f"Wrote analytics report to: {analytics_path}")
    if cfg.runtime.checkpoint_enabled:
        print(f"Wrote checkpoint to: {checkpoint_path}")
    print(f"Local records accepted: {local_records_accepted}")
    print(f"Cross-page records accepted: {cross_page_accepted}")
    print(f"Generated records (total): {local_records_accepted + cross_page_accepted}")
    print(f"Skipped pages: {skipped_records}")
    print(f"Estimated total API cost (USD): {round(total_estimated_cost_usd, 6)}")
    print(f"Average API latency (ms): {round(average_api_latency_ms, 2)}")


def main() -> None:
    """CLI entrypoint for pipeline execution."""
    parser = argparse.ArgumentParser(
        description="Multimodal PDF-to-ChatML QnA dataset pipeline."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config.yaml"),
        help="Path to YAML config file.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Process only 1 page per book for cheap validation.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from checkpoint file if available.",
    )
    parser.add_argument(
        "--skip-cross-page",
        action="store_true",
        help="Run Stage 1 only and skip cross-page synthesis stage.",
    )
    parser.add_argument(
        "--cross-page-only",
        action="store_true",
        help="Skip Stage 1 and run only cross-page synthesis from artifacts.",
    )
    args = parser.parse_args()
    run_pipeline(
        config_path=args.config,
        dry_run=args.dry_run,
        resume=args.resume,
        skip_cross_page=args.skip_cross_page,
        cross_page_only=args.cross_page_only,
    )

