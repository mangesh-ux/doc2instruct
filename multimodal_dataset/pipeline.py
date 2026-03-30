from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI
from tqdm import tqdm

from multimodal_dataset.chatml import append_jsonl, qa_to_chatml_record
from multimodal_dataset.config import AppConfig, load_config
from multimodal_dataset.openai_client import generate_qa_batch
from multimodal_dataset.pdf_pages import (
    iter_pdf_pages_as_data_urls,
    render_single_page_as_data_url,
)


UNUSABLE_STATUSES = {"blank", "unreadable"}


def _build_user_prompt(cfg: AppConfig, *, book_name: str, page_number: int) -> str:
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
    return {
        "system_prompt": cfg.prompts.system,
        "user_prompt": _build_user_prompt(cfg, book_name=book_name, page_number=page_number),
    }


def _iter_books(cfg: AppConfig) -> list[Path]:
    books_dir = cfg.input.books_dir
    if not books_dir.exists():
        raise FileNotFoundError(f"Books directory not found: {books_dir}")
    return sorted(books_dir.glob(cfg.input.glob))


def _record_skip(
    *,
    skipped_path: Path,
    book_name: str,
    page_number: int,
    status: str,
    reason: str,
    attempt_dpi: int,
) -> None:
    append_jsonl(
        skipped_path,
        {
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
    book_name: str,
    page_number: int,
    attempt: int,
    dpi: int,
    system_prompt: str,
    user_prompt: str,
) -> None:
    append_jsonl(
        prompt_log_path,
        {
            "timestamp_epoch": int(time.time()),
            "source_book": book_name,
            "source_page": page_number,
            "attempt": attempt,
            "dpi": dpi,
            "system_prompt": system_prompt,
            "user_prompt": user_prompt,
        },
    )


def _generate_with_retry(
    *,
    client: OpenAI,
    cfg: AppConfig,
    book_path: Path,
    book_name: str,
    page_number: int,
    initial_image_data_url: str,
) -> tuple[dict[str, Any], int]:
    preview = get_prompt_preview(cfg, book_name=book_name, page_number=page_number)
    prompt = preview["user_prompt"]
    system_prompt = preview["system_prompt"]
    attempt = 1
    if cfg.runtime.log_prompts:
        _log_prompt_request(
            prompt_log_path=cfg.runtime.prompt_log_path,
            book_name=book_name,
            page_number=page_number,
            attempt=attempt,
            dpi=cfg.runtime.dpi,
            system_prompt=system_prompt,
            user_prompt=prompt,
        )

    payload = generate_qa_batch(
        client=client,
        model=cfg.runtime.model,
        temperature=cfg.runtime.temperature,
        system_prompt=system_prompt,
        user_prompt=prompt,
        image_data_url=initial_image_data_url,
    )
    last_dpi = cfg.runtime.dpi

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
        if cfg.runtime.log_prompts:
            _log_prompt_request(
                prompt_log_path=cfg.runtime.prompt_log_path,
                book_name=book_name,
                page_number=page_number,
                attempt=attempt,
                dpi=cfg.runtime.retry_dpi,
                system_prompt=system_prompt,
                user_prompt=prompt,
            )
        payload = generate_qa_batch(
            client=client,
            model=cfg.runtime.model,
            temperature=cfg.runtime.temperature,
            system_prompt=system_prompt,
            user_prompt=prompt,
            image_data_url=retry_image_data_url,
        )

    return payload, last_dpi


def run_pipeline(config_path: Path, dry_run: bool) -> None:
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
    prompt_log_path = cfg.runtime.prompt_log_path
    if out_path.exists():
        out_path.unlink()
    if skipped_path.exists():
        skipped_path.unlink()
    if cfg.runtime.log_prompts and prompt_log_path.exists():
        prompt_log_path.unlink()

    client = OpenAI(api_key=api_key, timeout=cfg.runtime.request_timeout_seconds)
    written_records = 0
    skipped_records = 0

    for book in books:
        page_iter = iter_pdf_pages_as_data_urls(
            book,
            dpi=cfg.runtime.dpi,
            max_pages=1 if dry_run else cfg.runtime.max_pages_per_book,
        )
        for page in tqdm(page_iter, desc=f"Processing {book.name}"):
            payload, used_dpi = _generate_with_retry(
                client=client,
                cfg=cfg,
                book_path=book,
                book_name=book.name,
                page_number=page.page_number,
                initial_image_data_url=page.image_data_url,
            )
            status = payload.get("page_status", "unreadable")
            reason = payload.get("page_status_reason", "")
            items = payload.get("items", [])

            if status in UNUSABLE_STATUSES:
                _record_skip(
                    skipped_path=skipped_path,
                    book_name=book.name,
                    page_number=page.page_number,
                    status=status,
                    reason=reason,
                    attempt_dpi=used_dpi,
                )
                skipped_records += 1
                time.sleep(cfg.runtime.sleep_seconds_between_requests)
                continue

            for qa in items:
                record = qa_to_chatml_record(
                    qa_item=qa,
                    book_name=book.name,
                    page_number=page.page_number,
                    user_profile=cfg.dataset.user_profile,
                )
                append_jsonl(out_path, record)
                written_records += 1

            time.sleep(cfg.runtime.sleep_seconds_between_requests)

    print(f"Wrote dataset to: {out_path}")
    print(f"Wrote skipped-page log to: {skipped_path}")
    if cfg.runtime.log_prompts:
        print(f"Wrote prompt log to: {prompt_log_path}")
    print(f"Generated records: {written_records}")
    print(f"Skipped pages: {skipped_records}")


def main() -> None:
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
    args = parser.parse_args()
    run_pipeline(config_path=args.config, dry_run=args.dry_run)

