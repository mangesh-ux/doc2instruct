"""Convert reviewed candidates.csv into the final eval/custom/test.jsonl.

Usage:
    python scripts/finalize_eval_set.py
    python scripts/finalize_eval_set.py --in eval/custom/candidates.csv

Behavior:
    - Reads candidates.csv.
    - Keeps only rows with status == "accept" (or "edit" — treated as accepted).
    - Validates: question, answer, evidence_quote all non-empty.
    - Validates: pages parse as integers; cross_page rows have ≥2 pages.
    - Writes eval/custom/test.jsonl in a clean schema for the eval runner.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
from pathlib import Path

LOG_FORMAT = "%(asctime)s [%(levelname)s] %(message)s"

DEFAULT_IN = Path("eval/custom/candidates.csv")
DEFAULT_OUT = Path("eval/custom/test.jsonl")

ACCEPTED_STATUSES = {"accept", "edit"}


def _parse_pages(s: str) -> list[int]:
    return [int(p.strip()) for p in s.split(",") if p.strip()]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--in", dest="input_path", type=Path, default=DEFAULT_IN)
    parser.add_argument("--out", dest="output_path", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)

    if not args.input_path.exists():
        logging.error("Input not found: %s", args.input_path)
        return 1

    args.output_path.parent.mkdir(parents=True, exist_ok=True)

    kept = 0
    rejected = 0
    invalid = 0

    with args.input_path.open("r", encoding="utf-8") as fin, \
         args.output_path.open("w", encoding="utf-8") as fout:
        reader = csv.DictReader(fin)
        for i, row in enumerate(reader, start=2):  # start=2 accounts for header
            status = (row.get("status") or "").strip().lower()
            if status not in ACCEPTED_STATUSES:
                rejected += 1
                continue

            q = (row.get("candidate_question") or "").strip()
            a = (row.get("candidate_answer") or "").strip()
            ev = (row.get("evidence_quote") or "").strip()
            qtype = (row.get("question_type") or "").strip()

            if not q or not a:
                logging.warning("Row %d marked accept but missing q/a; skipping.", i)
                invalid += 1
                continue
            if not ev:
                logging.warning("Row %d has no evidence_quote; skipping.", i)
                invalid += 1
                continue

            try:
                pages = _parse_pages(row.get("pages", ""))
            except ValueError:
                logging.warning("Row %d has invalid pages: %r", i, row.get("pages"))
                invalid += 1
                continue

            if qtype == "cross_page" and len(pages) < 2:
                logging.warning("Row %d cross_page but <2 pages.", i)
                invalid += 1
                continue

            record = {
                "id": f"{row['paper_id']}_p{'-'.join(str(p) for p in pages)}_{kept:04d}",
                "paper_id": row["paper_id"],
                "arxiv_id": row.get("arxiv_id", row["paper_id"]),
                "question_type": qtype,
                "pages": pages,
                "question": q,
                "answer": a,
                "evidence_quote": ev,
            }
            fout.write(json.dumps(record) + "\n")
            kept += 1

    logging.info(
        "Finalized: kept=%d, rejected=%d, invalid=%d -> %s",
        kept, rejected, invalid, args.output_path,
    )
    if kept == 0:
        logging.warning("No rows accepted. Did you mark status=accept in the CSV?")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
