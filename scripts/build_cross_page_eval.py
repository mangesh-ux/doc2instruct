"""Build genuinely multi-hop cross-page eval candidates.

Why this exists
---------------
The single-quote candidate schema used by bootstrap_eval_set.py /
improve_candidates.py cannot express a multi-hop item: there is one
`evidence_quote` column, so a "cross_page" row is satisfied by a quote that
lives entirely on one page. Measured on the first build, 57/58 cross-page items
had all their evidence on a single page and 18 had answers fully covered by one
page's vocabulary — i.e. the subset could not measure Stage 2 at all.

This script asks for one verbatim span PER PAGE plus a justification, so the
downstream gate can verify each page really contributes evidence (the same
"supporting facts" idea HotpotQA uses).

Output columns add `evidence_quote_b` (span from the second page) and
`why_both_pages` to the usual candidate schema.

Usage:
    python scripts/build_cross_page_eval.py --limit 5      # smoke
    python scripts/build_cross_page_eval.py --workers 8    # full
"""

from __future__ import annotations

import argparse
import concurrent.futures
import csv
import json
import logging
import os
import sys
import threading
import time
from pathlib import Path

LOG_FORMAT = "%(asctime)s [%(levelname)s] %(message)s"

DEFAULT_IN_CSV = Path("eval/custom/candidates.csv")
DEFAULT_OUT_CSV = Path("eval/custom/cross_page_candidates.csv")
DEFAULT_PAGE_TEXTS = Path("eval/custom/page_texts.jsonl")
DEFAULT_MODEL = "gpt-4.1"
DEFAULT_WORKERS = 8
DEFAULT_CONTEXT_CHAR_LIMIT = 7000  # per page

OUT_FIELDS = [
    "paper_id", "arxiv_id", "pages", "question_type",
    "candidate_question", "candidate_answer",
    "evidence_quote", "evidence_quote_b", "why_both_pages",
    "status", "notes",
]

SYSTEM_PROMPT = """\
You write ONE genuinely multi-hop reading-comprehension question from TWO pages of a scientific paper.

The defining requirement: the question MUST be UNANSWERABLE from either page alone. A reader with only PAGE A, or only PAGE B, must be unable to produce the answer. The answer must require combining a fact from PAGE A with a distinct fact from PAGE B.

Typical valid shapes:
- A term/symbol/method is DEFINED on one page and USED or quantified on the other; the question asks for the combined consequence.
- One page states a condition or setup, the other states a result that depends on it; the question links them.
- One page gives a value, the other gives the rule that transforms it; the question asks for the transformed value.

Hard rules:
1. Forbidden: generic questions ("main contribution", "what does the paper study", "summarize").
2. The answer must be a specific, verifiable fact, SHORT (a phrase or one sentence, under 40 words), and never yes/no or true/false.
3. evidence_a MUST be copied CHARACTER-FOR-CHARACTER from PAGE A. evidence_b MUST be copied CHARACTER-FOR-CHARACTER from PAGE B. Do not paraphrase, do not shorten, do not join separate sentences with "...". Each must be one contiguous span. Re-read the page and confirm your span appears exactly.
4. evidence_a and evidence_b must BOTH be genuinely necessary. If either page's span is not needed to answer, the item is invalid.
5. why_both_pages: one sentence naming the fact taken from PAGE A and the fact taken from PAGE B.
6. If these two pages do not support any true multi-hop question (unrelated content, references list, one page is figures only), return status="reject" with a short reject_reason. Rejecting is expected and preferred over inventing a weak item.

Output strict JSON:
{
  "status": "accept" | "reject",
  "question": str,
  "answer": str,
  "evidence_a": str,
  "evidence_b": str,
  "why_both_pages": str,
  "reject_reason": str (only when status=reject)
}
No prose outside the JSON.
"""


def load_page_texts(path: Path) -> dict[tuple[str, int], str]:
    if not path.exists():
        raise FileNotFoundError(f"page_texts.jsonl not found at {path}")
    out: dict[tuple[str, int], str] = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            out[(str(row["paper_id"]), int(row["page"]))] = row["text"]
    return out


def parse_pages(value: str) -> list[int]:
    return [int(p.strip()) for p in str(value).split(",") if p.strip()]


def build_user_prompt(page_a: int, text_a: str, page_b: int, text_b: str,
                      char_limit: int) -> str:
    a = text_a[:char_limit]
    b = text_b[:char_limit]
    return (
        f"--- PAGE A (page {page_a}) ---\n{a}\n\n"
        f"--- PAGE B (page {page_b}) ---\n{b}\n\n"
        "Return JSON only."
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--in-csv", type=Path, default=DEFAULT_IN_CSV,
                        help="source of candidate page pairs (cross_page rows)")
    parser.add_argument("--out-csv", type=Path, default=DEFAULT_OUT_CSV)
    parser.add_argument("--page-texts", type=Path, default=DEFAULT_PAGE_TEXTS)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    parser.add_argument("--context-char-limit", type=int,
                        default=DEFAULT_CONTEXT_CHAR_LIMIT)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--enumerate-pairs", action="store_true",
                        help="ignore --in-csv and build page pairs directly from "
                             "page_texts.jsonl. Needed to grow the multi-hop pool: "
                             "~60%% of drafted items fail the single-page ablation, "
                             "so the pool must be several times the target size.")
    parser.add_argument("--max-gap", type=int, default=2,
                        help="with --enumerate-pairs, pair pages up to this far apart")
    parser.add_argument("--pairs-per-paper", type=int, default=20,
                        help="with --enumerate-pairs, cap pairs drawn per paper")
    parser.add_argument("--dry-run", action="store_true")
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

    page_texts = load_page_texts(args.page_texts)
    if args.enumerate_pairs:
        by_paper: dict[str, list[int]] = {}
        for (paper, page) in page_texts:
            by_paper.setdefault(paper, []).append(page)
        src_rows = []
        for paper in sorted(by_paper):
            pages = sorted(by_paper[paper])
            taken = 0
            # Nearer pages first: they share the most context and so give the
            # best chance of a real bridge between them.
            for gap in range(1, max(1, args.max_gap) + 1):
                for p in pages:
                    if p + gap not in by_paper[paper]:
                        continue
                    if taken >= args.pairs_per_paper:
                        break
                    src_rows.append({"paper_id": paper, "arxiv_id": paper,
                                     "pages": f"{p},{p + gap}",
                                     "question_type": "cross_page"})
                    taken += 1
                if taken >= args.pairs_per_paper:
                    break
        logging.info("Enumerated %d page-pairs across %d papers (max_gap=%d).",
                     len(src_rows), len(by_paper), args.max_gap)
    else:
        with args.in_csv.open("r", encoding="utf-8") as f:
            src_rows = [r for r in csv.DictReader(f)
                        if (r.get("question_type") or "") == "cross_page"]
        logging.info("Found %d cross_page page-pairs in %s", len(src_rows), args.in_csv)

    # Keep only pairs where both pages have usable extracted text.
    tasks = []
    for row in src_rows:
        paper = str(row.get("paper_id", ""))
        try:
            pages = parse_pages(row.get("pages", ""))
        except ValueError:
            continue
        if len(pages) < 2:
            continue
        page_a, page_b = pages[0], pages[1]
        text_a = page_texts.get((paper, page_a), "")
        text_b = page_texts.get((paper, page_b), "")
        if len(text_a.strip()) < 400 or len(text_b.strip()) < 400:
            continue
        tasks.append((row, paper, page_a, text_a, page_b, text_b))
    if args.limit is not None:
        tasks = tasks[: args.limit]
    logging.info("Generating %d multi-hop candidates with %s (%d workers).",
                 len(tasks), args.model, args.workers)

    if args.dry_run:
        row, paper, pa, ta, pb, tb = tasks[0]
        print("=== SYSTEM ===")
        print(SYSTEM_PROMPT)
        print("=== USER (truncated) ===")
        print(build_user_prompt(pa, ta, pb, tb, 600)[:1800])
        return 0

    from openai import OpenAI  # type: ignore

    results: list[dict] = []
    counters = {"accept": 0, "reject": 0, "errors": 0, "done": 0}
    lock = threading.Lock()
    t0 = time.time()

    def work(task):
        row, paper, page_a, text_a, page_b, text_b = task
        client = OpenAI()
        user = build_user_prompt(page_a, text_a, page_b, text_b,
                                 args.context_char_limit)
        try:
            resp = client.chat.completions.create(
                model=args.model,
                messages=[{"role": "system", "content": SYSTEM_PROMPT},
                          {"role": "user", "content": user}],
                response_format={"type": "json_object"},
                temperature=0.2,
            )
            payload = json.loads(resp.choices[0].message.content)
        except Exception as exc:  # noqa: BLE001
            return task, None, str(exc)
        return task, payload, None

    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        futures = [pool.submit(work, t) for t in tasks]
        for fut in concurrent.futures.as_completed(futures):
            try:
                task, payload, err = fut.result()
            except Exception as exc:  # noqa: BLE001
                with lock:
                    counters["errors"] += 1
                logging.warning("worker crashed: %s", exc)
                continue
            row, paper, page_a, text_a, page_b, text_b = task
            with lock:
                counters["done"] += 1
                if err or payload is None:
                    counters["errors"] += 1
                    logging.warning("%s p%s/%s failed: %s", paper, page_a, page_b, err)
                elif (payload.get("status") or "accept").lower() == "reject":
                    counters["reject"] += 1
                else:
                    results.append({
                        "paper_id": paper,
                        "arxiv_id": row.get("arxiv_id", paper),
                        "pages": f"{page_a},{page_b}",
                        "question_type": "cross_page",
                        "candidate_question": (payload.get("question") or "").strip(),
                        "candidate_answer": (payload.get("answer") or "").strip(),
                        "evidence_quote": (payload.get("evidence_a") or "").strip(),
                        "evidence_quote_b": (payload.get("evidence_b") or "").strip(),
                        "why_both_pages": (payload.get("why_both_pages") or "").strip(),
                        "status": "pending",
                        "notes": "multihop-v1",
                    })
                    counters["accept"] += 1
                if counters["done"] % 25 == 0:
                    logging.info("  [%d/%d] accept=%d reject=%d errors=%d (%.1fs)",
                                 counters["done"], len(tasks), counters["accept"],
                                 counters["reject"], counters["errors"],
                                 time.time() - t0)

    results.sort(key=lambda r: (r["paper_id"], parse_pages(r["pages"])))
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.out_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=OUT_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(results)

    logging.info("Done. accept=%d model_reject=%d errors=%d elapsed=%.1fs -> %s",
                 counters["accept"], counters["reject"], counters["errors"],
                 time.time() - t0, args.out_csv)
    logging.info("Next: python scripts/verify_candidates.py "
                 "--in-csv %s eval/custom/candidates_v3.csv", args.out_csv)
    return 0


if __name__ == "__main__":
    sys.exit(main())
