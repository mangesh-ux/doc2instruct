"""Rewrite weak LLM-drafted candidates with a tighter rubric.

Takes the existing eval/custom/candidates.csv produced by bootstrap_eval_set.py
and rewrites each `pending` row using a much stricter QA-generation prompt
than the default bootstrap. Writes to candidates_v2.csv (default) so the
original is preserved.

Usage:
    # smoke (cheap — 5 rows only)
    python scripts/improve_candidates.py --limit 5

    # full run
    python scripts/improve_candidates.py

    # in-place rewrite (backs up original first)
    python scripts/improve_candidates.py --in-place

What changes vs the original bootstrap LLM prompt:
    - Forbids generic "main focus / main contribution" questions.
    - Requires the answer to be a specific verifiable fact: a quantity, a
      definition, a named mechanism, an enumerated list, etc.
    - Requires the evidence_quote to be DISTINCT from the answer (supporting
      context, not a verbatim copy of the answer sentence).
    - Asks the model to choose a question_type:
        definition | quantitative | mechanism | example_reasoning | comparison
      and stamps it in the `notes` column so you can balance types when reviewing.
    - For cross_page rows: requires that answering needs info from BOTH pages.
      If the model judges it doesn't, it returns reject_reason and the row is
      marked status=reject (you don't waste eyeball time on dud cross-page items).
    - If no good question exists on the page (figure-only, references, etc.),
      the model returns status=reject.

Provenance: every rewritten row gets `notes` starting with `llm-improved-v2:`
so you can grep / filter when spot-checking.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import csv
import json
import logging
import os
import shutil
import sys
import threading
import time
from pathlib import Path

LOG_FORMAT = "%(asctime)s [%(levelname)s] %(message)s"

DEFAULT_IN_CSV = Path("eval/custom/candidates.csv")
DEFAULT_OUT_CSV = Path("eval/custom/candidates_v2.csv")
DEFAULT_PAGE_TEXTS = Path("eval/custom/page_texts.jsonl")
DEFAULT_MODEL = "gpt-4.1"
DEFAULT_CONTEXT_CHAR_LIMIT = 8000  # full page text fed to the model
DEFAULT_WORKERS = 8

VALID_TYPES = {"definition", "quantitative", "mechanism",
               "example_reasoning", "comparison"}


SYSTEM_PROMPT = """\
You write ONE high-quality grounded reading-comprehension QA pair from a passage of a scientific paper.

Hard rules:
1. The question MUST require having read this specific page. Forbidden: "main focus", "main contribution", "what does the paper study", "what is the purpose" — these are too generic.
2. The answer MUST be a specific, verifiable fact. Acceptable: a defined term, a quantity / formula / identity, a named mechanism, an enumerated set, a concrete cause-effect statement. NOT acceptable: vague paraphrases or summaries. Keep the answer SHORT and self-contained — a phrase or at most one sentence (under 40 words), so it can be scored by exact match. The answer must NEVER be yes/no or true/false, and the question must not be answerable by guessing.
3. The evidence_quote MUST be COPIED CHARACTER-FOR-CHARACTER from the passage. Select one or two consecutive sentences and reproduce them EXACTLY as they appear — do not paraphrase, summarize, shorten, fix grammar, or join non-adjacent fragments. This is a copy-paste operation, not a writing task. It is fine and expected for the quote to contain the answer. Before answering, re-read your quote and confirm the exact string appears in the passage.
4. Pick a question_type from: definition, quantitative, mechanism, example_reasoning, comparison.
5. If the passage is poor quality for QA (figure-only, references list, acknowledgements, math without enough text), respond with status="reject" and a short reject_reason.
6. For cross_page items (multiple pages provided), the answer MUST require info from BOTH pages. Otherwise return status="reject" with reject_reason="not truly cross-page".

Output strict JSON with these keys:
{
  "status": "accept" | "reject",
  "question_type": one of the five tags above (omit on reject),
  "question": str (omit on reject),
  "answer": str (omit on reject),
  "evidence_quote": str (verbatim from passage; omit on reject),
  "reject_reason": str (only when status=reject)
}
No prose outside the JSON.
"""


def load_page_texts(path: Path) -> dict[tuple[str, int], str]:
    out: dict[tuple[str, int], str] = {}
    if not path.exists():
        raise FileNotFoundError(
            f"page_texts.jsonl not found at {path}. "
            "Run bootstrap_eval_set.py first."
        )
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            out[(row["paper_id"], int(row["page"]))] = row["text"]
    return out


def build_passage(row: dict, page_texts: dict[tuple[str, int], str],
                  char_limit: int) -> tuple[str, list[int]]:
    pages = [int(p.strip()) for p in row["pages"].split(",") if p.strip()]
    parts = []
    for p in pages:
        text = page_texts.get((row["paper_id"], p), "")
        parts.append(f"--- page {p} ---\n{text}")
    full = "\n\n".join(parts)
    if len(full) > char_limit:
        full = full[:char_limit] + "\n[...truncated]"
    return full, pages


def rewrite_one(row: dict, passage: str, qtype_hint: str,
                client, model: str) -> dict:
    user = (
        f"question_type_hint: {qtype_hint}\n"
        f"passage:\n{passage}\n\n"
        "Return JSON only."
    )
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "system", "content": SYSTEM_PROMPT},
                  {"role": "user", "content": user}],
        response_format={"type": "json_object"},
        temperature=0.2,
    )
    return json.loads(resp.choices[0].message.content)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--in-csv", type=Path, default=DEFAULT_IN_CSV)
    parser.add_argument("--out-csv", type=Path, default=DEFAULT_OUT_CSV)
    parser.add_argument("--page-texts", type=Path, default=DEFAULT_PAGE_TEXTS)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--context-char-limit", type=int,
                        default=DEFAULT_CONTEXT_CHAR_LIMIT)
    parser.add_argument("--limit", type=int, default=None,
                        help="cap how many rows to rewrite (smoke runs)")
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS,
                        help=f"parallel API workers (default {DEFAULT_WORKERS})")
    parser.add_argument("--all-rows", dest="only_pending", action="store_false",
                        default=True,
                        help="rewrite every row, including ones already reviewed "
                             "(default: only status=pending rows are touched)")
    parser.add_argument("--in-place", action="store_true",
                        help="overwrite candidates.csv after backing it up")
    parser.add_argument("--dry-run", action="store_true",
                        help="print prompt for first row and exit (no API calls)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)

    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass
    if not args.dry_run and not os.getenv("OPENAI_API_KEY"):
        logging.error("OPENAI_API_KEY not set in env/.env. Aborting.")
        return 1

    if not args.in_csv.exists():
        logging.error("Input CSV not found: %s", args.in_csv)
        return 1

    page_texts = load_page_texts(args.page_texts)
    logging.info("Loaded page texts for %d (paper, page) pairs.", len(page_texts))

    with args.in_csv.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        rows = list(reader)
    logging.info("Loaded %d candidate rows from %s.", len(rows), args.in_csv)

    if args.dry_run:
        # Print the prompt for the first pending row and quit.
        pending = [r for r in rows if (r.get("status") or "").lower() == "pending"]
        if not pending:
            logging.error("No pending rows to dry-run on.")
            return 1
        sample = pending[0]
        passage, pages = build_passage(sample, page_texts, args.context_char_limit)
        qtype_hint = ("cross_page (requires both pages)"
                      if sample.get("question_type") == "cross_page"
                      else "single_page")
        print("=== SYSTEM PROMPT ===")
        print(SYSTEM_PROMPT)
        print("\n=== USER PROMPT (sample) ===")
        print(f"question_type_hint: {qtype_hint}")
        print(f"passage (first 800 chars):\n{passage[:800]}...")
        print(f"\n(passage length: {len(passage)} chars, pages: {pages})")
        return 0

    from openai import OpenAI  # type: ignore

    # Pick the rows to work on up front so --limit is deterministic and the
    # parallel pass can't race on selection.
    targets: list[int] = []
    skipped_already_reviewed = 0
    for idx, row in enumerate(rows):
        status = (row.get("status") or "").lower()
        if args.only_pending and status != "pending":
            skipped_already_reviewed += 1
            continue
        targets.append(idx)
    if args.limit is not None:
        targets = targets[: args.limit]

    logging.info("Rewriting %d rows with %s using %d workers.",
                 len(targets), args.model, args.workers)

    counters = {"rewritten": 0, "rejected": 0, "errors": 0, "done": 0}
    lock = threading.Lock()
    t0 = time.time()

    def work(idx: int) -> tuple[int, dict | None, str | None]:
        """Rewrite one row. Returns (idx, payload, error). Thread-safe."""
        row = rows[idx]
        # One client per thread: the SDK client is not guaranteed safe to share
        # across threads under load.
        client = OpenAI()
        try:
            passage, _pages = build_passage(row, page_texts, args.context_char_limit)
        except Exception as exc:  # noqa: BLE001
            return idx, None, f"passage build failed: {exc}"
        qtype_hint = ("cross_page (answer must require info from BOTH pages)"
                      if row.get("question_type") == "cross_page"
                      else "single_page")
        try:
            payload = rewrite_one(row, passage, qtype_hint, client, args.model)
        except Exception as exc:  # noqa: BLE001
            return idx, None, f"rewrite failed: {exc}"
        return idx, payload, None

    def apply_result(idx: int, payload: dict | None, error: str | None) -> None:
        row = rows[idx]
        if error is not None or payload is None:
            logging.warning("Row %d (%s pages=%s): %s",
                            idx + 2, row.get("paper_id"), row.get("pages"), error)
            counters["errors"] += 1
            return
        if (payload.get("status") or "accept").lower() == "reject":
            row["status"] = "reject"
            row["notes"] = f"llm-improved-v2:reject - {payload.get('reject_reason', '')}"
            row["candidate_question"] = ""
            row["candidate_answer"] = ""
            row["evidence_quote"] = ""
            counters["rejected"] += 1
        else:
            qtype_tag = payload.get("question_type", "")
            if qtype_tag not in VALID_TYPES:
                qtype_tag = "unspecified"
            row["candidate_question"] = (payload.get("question") or "").strip()
            row["candidate_answer"] = (payload.get("answer") or "").strip()
            row["evidence_quote"] = (payload.get("evidence_quote") or "").strip()
            # status stays "pending" — a human still accepts it.
            row["status"] = "pending"
            row["notes"] = f"llm-improved-v2:{qtype_tag}"
            counters["rewritten"] += 1

    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        futures = {pool.submit(work, idx): idx for idx in targets}
        for future in concurrent.futures.as_completed(futures):
            try:
                idx, payload, error = future.result()
            except Exception as exc:  # noqa: BLE001
                logging.warning("Worker crashed: %s", exc)
                with lock:
                    counters["errors"] += 1
                continue
            with lock:
                apply_result(idx, payload, error)
                counters["done"] += 1
                if counters["done"] % 25 == 0:
                    logging.info(
                        "  [%d/%d] rewritten=%d rejected=%d errors=%d (%.1fs)",
                        counters["done"], len(targets), counters["rewritten"],
                        counters["rejected"], counters["errors"], time.time() - t0,
                    )

    rewritten = counters["rewritten"]
    rejected_by_model = counters["rejected"]
    errors = counters["errors"]

    # Write output.
    out_path = args.in_csv if args.in_place else args.out_csv
    if args.in_place:
        backup = args.in_csv.with_suffix(".csv.bak")
        shutil.copy2(args.in_csv, backup)
        logging.info("In-place mode: backed up original to %s", backup)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    logging.info(
        "Done. rewritten=%d rejected_by_model=%d skipped_already_reviewed=%d "
        "errors=%d elapsed=%.1fs -> %s",
        rewritten, rejected_by_model, skipped_already_reviewed,
        errors, time.time() - t0, out_path,
    )
    logging.info(
        "Next: open %s in the review notebook. Rows with notes starting "
        "'llm-improved-v2:' are AI-rewritten — spot-check ~30 across question "
        "types and accept in bulk.",
        out_path,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
