"""Prove cross-page items are really multi-hop, by single-page ablation.

An LLM asked "does this need both pages?" will say yes far more often than is
true (measured: it accepted items whose answer sat entirely inside one page's
quote). So instead of trusting that judgment, this script tests it.

For each cross-page candidate:
  1. Ask a strong model to answer the question given ONLY page A.
  2. Ask again given ONLY page B.
  3. Score each single-page attempt against the gold answer (token F1, plus an
     explicit INSUFFICIENT escape hatch the model is told to use).

An item is kept only if NEITHER single page yields the answer. That is the same
shortcut audit people run on multi-hop QA benchmarks, and it turns "we asked the
model to make it multi-hop" into "no single page suffices, and here are the
numbers".

Columns added: ablation_f1_a, ablation_f1_b, ablation_verdict.
Rows that fail get status=reject with the reason recorded, so nothing is
silently dropped.

Usage:
    python scripts/ablate_cross_page.py --limit 5        # smoke
    python scripts/ablate_cross_page.py --workers 8      # full
"""

from __future__ import annotations

import argparse
import collections
import concurrent.futures
import csv
import json
import logging
import os
import re
import string
import sys
import threading
import time
from pathlib import Path

LOG_FORMAT = "%(asctime)s [%(levelname)s] %(message)s"

DEFAULT_IN_CSV = Path("eval/custom/cross_page_candidates.csv")
DEFAULT_OUT_CSV = Path("eval/custom/cross_page_ablated.csv")
DEFAULT_PAGE_TEXTS = Path("eval/custom/page_texts.jsonl")
DEFAULT_REPORT = Path("eval/custom/ablation_report.json")
DEFAULT_MODEL = "gpt-4.1"
DEFAULT_WORKERS = 8
DEFAULT_CONTEXT_CHAR_LIMIT = 7000
# Above this token-F1 against the gold answer, a single page is treated as
# sufficient and the item is not multi-hop.
DEFAULT_F1_THRESHOLD = 0.6

INSUFFICIENT = "INSUFFICIENT"

SYSTEM_PROMPT = f"""\
You answer a question using ONLY the single page of a scientific paper provided.

Rules:
- If the page contains enough information, answer concisely (a phrase or one sentence).
- If the page does NOT contain enough information to answer, reply with exactly: {INSUFFICIENT}
- Never guess, never use outside knowledge, never explain. Output only the answer or {INSUFFICIENT}.
"""


def _normalize(text: str) -> str:
    """SQuAD-style normalization for token F1."""
    text = (text or "").lower()
    text = "".join(ch for ch in text if ch not in set(string.punctuation))
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


def token_f1(pred: str, gold: str) -> float:
    pred_tokens = _normalize(pred).split()
    gold_tokens = _normalize(gold).split()
    if not pred_tokens or not gold_tokens:
        return float(pred_tokens == gold_tokens)
    common = collections.Counter(pred_tokens) & collections.Counter(gold_tokens)
    overlap = sum(common.values())
    if overlap == 0:
        return 0.0
    precision = overlap / len(pred_tokens)
    recall = overlap / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def load_page_texts(path: Path) -> dict[tuple[str, int], str]:
    out: dict[tuple[str, int], str] = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                row = json.loads(line)
                out[(str(row["paper_id"]), int(row["page"]))] = row["text"]
    return out


def parse_pages(value: str) -> list[int]:
    return [int(p.strip()) for p in str(value).split(",") if p.strip()]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--in-csv", type=Path, default=DEFAULT_IN_CSV)
    parser.add_argument("--out-csv", type=Path, default=DEFAULT_OUT_CSV)
    parser.add_argument("--page-texts", type=Path, default=DEFAULT_PAGE_TEXTS)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    parser.add_argument("--context-char-limit", type=int,
                        default=DEFAULT_CONTEXT_CHAR_LIMIT)
    parser.add_argument("--f1-threshold", type=float, default=DEFAULT_F1_THRESHOLD)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)

    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass
    if not os.getenv("OPENAI_API_KEY"):
        logging.error("OPENAI_API_KEY not set. Aborting.")
        return 1

    if not args.in_csv.exists():
        logging.error("Input not found: %s. Run build_cross_page_eval.py first.",
                      args.in_csv)
        return 1

    page_texts = load_page_texts(args.page_texts)
    with args.in_csv.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames or [])
        rows = list(reader)
    for extra in ("ablation_f1_a", "ablation_f1_b", "ablation_verdict"):
        if extra not in fieldnames:
            fieldnames.append(extra)

    targets = list(range(len(rows)))
    if args.limit is not None:
        targets = targets[: args.limit]
    logging.info("Ablating %d cross-page items with %s (%d workers).",
                 len(targets), args.model, args.workers)

    from openai import OpenAI  # type: ignore

    counters = {"kept": 0, "shortcut": 0, "errors": 0, "done": 0}
    lock = threading.Lock()
    t0 = time.time()

    def ask_single_page(client, question: str, page_text: str) -> str:
        resp = client.chat.completions.create(
            model=args.model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content":
                    f"PAGE:\n{page_text[:args.context_char_limit]}\n\n"
                    f"QUESTION: {question}\n\nAnswer or {INSUFFICIENT}:"},
            ],
            temperature=0.0,
        )
        return (resp.choices[0].message.content or "").strip()

    def work(idx: int):
        row = rows[idx]
        paper = str(row.get("paper_id", ""))
        try:
            pages = parse_pages(row.get("pages", ""))
        except ValueError:
            return idx, None, "bad pages"
        if len(pages) < 2:
            return idx, None, "fewer than 2 pages"
        question = row.get("candidate_question", "")
        gold = row.get("candidate_answer", "")
        client = OpenAI()
        out = {}
        for tag, page in (("a", pages[0]), ("b", pages[1])):
            text = page_texts.get((paper, page), "")
            if not text:
                out[tag] = (INSUFFICIENT, 0.0)
                continue
            try:
                answer = ask_single_page(client, question, text)
            except Exception as exc:  # noqa: BLE001
                return idx, None, f"api error: {exc}"
            f1 = 0.0 if answer.upper().startswith(INSUFFICIENT) else token_f1(answer, gold)
            out[tag] = (answer, f1)
        return idx, out, None

    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        futures = [pool.submit(work, i) for i in targets]
        for fut in concurrent.futures.as_completed(futures):
            try:
                idx, out, err = fut.result()
            except Exception as exc:  # noqa: BLE001
                with lock:
                    counters["errors"] += 1
                logging.warning("worker crashed: %s", exc)
                continue
            row = rows[idx]
            with lock:
                counters["done"] += 1
                if err or out is None:
                    counters["errors"] += 1
                    row["status"] = "reject"
                    row["ablation_verdict"] = f"error: {err}"
                else:
                    f1a = out["a"][1]
                    f1b = out["b"][1]
                    row["ablation_f1_a"] = f"{f1a:.3f}"
                    row["ablation_f1_b"] = f"{f1b:.3f}"
                    if max(f1a, f1b) >= args.f1_threshold:
                        which = "A" if f1a >= f1b else "B"
                        row["status"] = "reject"
                        row["ablation_verdict"] = (
                            f"single_page_sufficient(page_{which},f1={max(f1a, f1b):.2f})")
                        counters["shortcut"] += 1
                    else:
                        row["status"] = "pending"
                        row["ablation_verdict"] = "needs_both_pages"
                        counters["kept"] += 1
                if counters["done"] % 25 == 0:
                    logging.info("  [%d/%d] needs_both=%d shortcut=%d errors=%d (%.1fs)",
                                 counters["done"], len(targets), counters["kept"],
                                 counters["shortcut"], counters["errors"],
                                 time.time() - t0)

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.out_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    kept_rows = [r for r in rows if r.get("ablation_verdict") == "needs_both_pages"]
    report = {
        "input_csv": str(args.in_csv),
        "output_csv": str(args.out_csv),
        "model": args.model,
        "f1_threshold": args.f1_threshold,
        "items_tested": len(targets),
        "needs_both_pages": counters["kept"],
        "rejected_single_page_sufficient": counters["shortcut"],
        "errors": counters["errors"],
        "shortcut_rate": round(counters["shortcut"] / max(1, len(targets)), 4),
        "distinct_papers_kept": len({r.get("paper_id") for r in kept_rows}),
        "method": (
            "Each question was answered by the model given only page A, then only "
            "page B. Token F1 against the gold answer; an item is kept only if "
            "both single-page attempts score below the threshold, i.e. no single "
            "page is sufficient."
        ),
    }
    with args.report.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    logging.info("=" * 62)
    logging.info("needs_both_pages=%d  single_page_sufficient=%d  errors=%d",
                 counters["kept"], counters["shortcut"], counters["errors"])
    logging.info("shortcut rate: %.1f%% (this is what the ablation caught)",
                 100 * counters["shortcut"] / max(1, len(targets)))
    logging.info("Wrote %s and %s", args.out_csv, args.report)
    return 0


if __name__ == "__main__":
    sys.exit(main())
