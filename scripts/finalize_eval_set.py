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
    - Carries per-item provenance (evidence coverage, which checks passed, which
      generation the item came from) so a reader can audit any single item, and
      writes a sidecar test_provenance.json describing how the set was built.

Provenance honesty: `review` is "machine_gate" unless the row was touched by a
human in the review notebook (status "edit", or notes containing
"human-reviewed"). Do not describe the set as hand-verified until those rows
say so.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
from pathlib import Path

LOG_FORMAT = "%(asctime)s [%(levelname)s] %(message)s"

DEFAULT_IN = Path("eval/custom/candidates_verified.csv")
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
    human_reviewed = 0
    type_counts: dict[str, int] = {}
    papers: set[str] = set()

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
            ev_b = (row.get("evidence_quote_b") or "").strip()
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

            notes = (row.get("notes") or "").strip()
            human_touched = status == "edit" or "human-reviewed" in notes.lower()
            if human_touched:
                human_reviewed += 1

            record = {
                "id": f"{row['paper_id']}_p{'-'.join(str(p) for p in pages)}_{kept:04d}",
                "paper_id": row["paper_id"],
                "arxiv_id": row.get("arxiv_id", row["paper_id"]),
                "question_type": qtype,
                "pages": pages,
                "question": q,
                "answer": a,
                "evidence_quote": ev,
                "provenance": {
                    "multi_hop_verified": row.get("ablation_verdict", ""),
                    "single_page_f1": {
                        "page_a": row.get("ablation_f1_a", ""),
                        "page_b": row.get("ablation_f1_b", ""),
                    } if row.get("ablation_verdict") else None,
                    "review": "human" if human_touched else "machine_gate",
                    "checks_passed": row.get("verify_reasons", ""),
                    "evidence_coverage": float(row["quote_coverage"])
                    if row.get("quote_coverage") else None,
                    "drafted_by": notes,
                    "source_csv": row.get("source_csv", ""),
                },
            }
            # Multi-hop items carry one span per page (the "supporting facts"),
            # so a grader can see which page contributed what.
            if ev_b:
                record["evidence_quotes"] = [
                    {"page": pages[0], "quote": ev},
                    {"page": pages[1], "quote": ev_b},
                ]
                record["why_both_pages"] = row.get("why_both_pages", "")
            fout.write(json.dumps(record, ensure_ascii=False) + "\n")
            kept += 1
            type_counts[qtype] = type_counts.get(qtype, 0) + 1
            papers.add(row["paper_id"])

    logging.info(
        "Finalized: kept=%d, rejected=%d, invalid=%d -> %s",
        kept, rejected, invalid, args.output_path,
    )
    if kept == 0:
        logging.warning("No rows accepted. Did you mark status=accept in the CSV?")
        return 1

    n_single = type_counts.get("single_page", 0)
    n_cross = type_counts.get("cross_page", 0)
    provenance = {
        "source_csv": str(args.input_path),
        "output_jsonl": str(args.output_path),
        "total_items": kept,
        "single_page_items": n_single,
        "cross_page_items": n_cross,
        "cross_over_single_ratio": round(n_cross / n_single, 3) if n_single else None,
        "distinct_papers": len(papers),
        "human_reviewed_items": human_reviewed,
        "machine_gate_items": kept - human_reviewed,
        "pipeline": [
            "bootstrap_eval_set.py: page-text extraction + LLM-drafted candidates",
            "improve_candidates.py: rewrite under strict rubric (specific, verifiable, verbatim evidence)",
            "build_cross_page_eval.py: one verbatim span per page + why_both_pages",
            "ablate_cross_page.py: keep only items neither page A nor page B can answer",
            "verify_candidates.py: mechanical gate (verbatim evidence coverage, "
            "generic-question filter, yes/no filter, dedup, per-page grounding) "
            "+ quota-balanced selection across papers",
            "finalize_eval_set.py: schema + provenance",
        ],
    }
    prov_path = args.output_path.with_name(
        args.output_path.stem + "_provenance.json")
    with prov_path.open("w", encoding="utf-8") as f:
        json.dump(provenance, f, indent=2, ensure_ascii=False)

    logging.info("single_page=%d cross_page=%d ratio=%s papers=%d",
                 n_single, n_cross,
                 f"{n_cross / n_single:.2f}" if n_single else "n/a", len(papers))
    if human_reviewed == 0:
        logging.warning(
            "No item is human-reviewed. This set is MACHINE-VERIFIED only - do "
            "not publish it as hand-verified until you spot-check in the notebook."
        )
    logging.info("Wrote provenance to %s", prov_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
