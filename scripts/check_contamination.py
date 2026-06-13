"""Contamination gate: verify held-out papers did not leak into training data.

Implements eval_plan.md section 4 (anti-contamination protocol). Run this AFTER
doc2instruct has generated the training dataset and BEFORE any fine-tuning.

Two independent checks, using corpus/manifest.jsonl as the authority on splits:

  1. ID check     — no held-out arxiv_id / paper id appears in the generated
                    dataset's record metadata (source_book / pack_id).
  2. Text check   — no held-out paper's page text overlaps the generated dataset
                    above a threshold, measured with shingled (k-gram) hashing.

Exit codes:
    0  clean — safe to fine-tune.
    2  contamination detected — STOP.
    1  usage / IO error.

Usage:
    python scripts/check_contamination.py
    python scripts/check_contamination.py \\
        --dataset output/chatml_dataset.jsonl \\
        --manifest corpus/manifest.jsonl \\
        --holdout-page-texts eval/custom/page_texts.jsonl \\
        --shingle-size 8 --overlap-threshold 0.02

Notes:
    - The text check needs holdout page text. The cheapest source is
      eval/custom/page_texts.jsonl (produced by bootstrap_eval_set.py). If it is
      missing, the script extracts text directly from corpus/holdout/*.pdf.
    - "Overlap" is the fraction of a held-out paper's shingles that also appear
      anywhere in the generated dataset text. A tiny non-zero value is normal
      (shared boilerplate / common phrases); the threshold guards against real
      leakage. Tune --overlap-threshold if your domain is very templated.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import re
import sys
from pathlib import Path

LOG_FORMAT = "%(asctime)s [%(levelname)s] %(message)s"

DEFAULT_DATASET = Path("output/chatml_dataset.jsonl")
DEFAULT_MANIFEST = Path("corpus/manifest.jsonl")
DEFAULT_HOLDOUT_PAGE_TEXTS = Path("eval/custom/page_texts.jsonl")
DEFAULT_HOLDOUT_PDF_DIR = Path("corpus/holdout")
DEFAULT_SHINGLE_SIZE = 8       # words per shingle
DEFAULT_OVERLAP_THRESHOLD = 0.02  # >2% of a paper's shingles leaking = fail
DEFAULT_MIN_SHINGLES = 50      # skip papers with too little text to judge


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower()).strip()


def _word_shingles(text: str, k: int) -> set[int]:
    """Return a set of hashed k-word shingles for fast overlap estimation."""
    words = _normalize(text).split()
    if len(words) < k:
        return set()
    out: set[int] = set()
    for i in range(len(words) - k + 1):
        gram = " ".join(words[i : i + k])
        out.add(int(hashlib.blake2b(gram.encode("utf-8"), digest_size=8).hexdigest(), 16))
    return out


def load_manifest_splits(manifest_path: Path) -> tuple[set[str], set[str]]:
    """Return (holdout_ids, train_ids) from the combined manifest."""
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Manifest not found: {manifest_path}. Run split_corpus.py first."
        )
    holdout: set[str] = set()
    train: set[str] = set()
    with manifest_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            arxiv_id = str(row.get("arxiv_id", ""))
            split = str(row.get("split", ""))
            if not arxiv_id:
                continue
            if split == "holdout":
                holdout.add(arxiv_id)
            elif split == "train":
                train.add(arxiv_id)
    return holdout, train


def iter_dataset_records(dataset_path: Path):
    with dataset_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def dataset_text_and_sources(dataset_path: Path) -> tuple[str, set[str]]:
    """Collect all assistant/user text and all source identifiers in the dataset."""
    text_parts: list[str] = []
    sources: set[str] = set()
    for rec in iter_dataset_records(dataset_path):
        for msg in rec.get("messages", []):
            if msg.get("role") in {"user", "assistant"}:
                text_parts.append(str(msg.get("content", "")))
        meta = rec.get("metadata", {}) or {}
        for key in ("source_book", "pack_id"):
            val = str(meta.get(key, ""))
            if val:
                sources.add(val)
    return "\n".join(text_parts), sources


def load_holdout_texts(
    holdout_ids: set[str],
    page_texts_path: Path,
    holdout_pdf_dir: Path,
) -> dict[str, str]:
    """Map holdout paper id -> concatenated text, preferring cached page texts."""
    texts: dict[str, list[str]] = {pid: [] for pid in holdout_ids}

    if page_texts_path.exists():
        with page_texts_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                pid = str(row.get("paper_id", ""))
                if pid in texts:
                    texts[pid].append(str(row.get("text", "")))
        logging.info("Loaded holdout text from cache: %s", page_texts_path)

    # Fall back to PDFs for any holdout paper we have no cached text for.
    missing = [pid for pid, parts in texts.items() if not any(parts)]
    if missing:
        try:
            import fitz  # PyMuPDF
        except ImportError:
            logging.warning(
                "No cached text for %d holdout papers and PyMuPDF not installed; "
                "those papers will be skipped in the text check: %s",
                len(missing), missing,
            )
        else:
            for pid in missing:
                pdf = holdout_pdf_dir / f"{pid}.pdf"
                if not pdf.exists():
                    logging.warning("Holdout PDF missing for %s: %s", pid, pdf)
                    continue
                with fitz.open(pdf) as doc:
                    texts[pid] = [page.get_text("text") or "" for page in doc]

    return {pid: "\n".join(parts) for pid, parts in texts.items()}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET,
                        help="generated training dataset JSONL to audit")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--holdout-page-texts", type=Path,
                        default=DEFAULT_HOLDOUT_PAGE_TEXTS)
    parser.add_argument("--holdout-pdf-dir", type=Path,
                        default=DEFAULT_HOLDOUT_PDF_DIR)
    parser.add_argument("--shingle-size", type=int, default=DEFAULT_SHINGLE_SIZE)
    parser.add_argument("--overlap-threshold", type=float,
                        default=DEFAULT_OVERLAP_THRESHOLD)
    parser.add_argument("--min-shingles", type=int, default=DEFAULT_MIN_SHINGLES)
    parser.add_argument("--report", type=Path,
                        default=Path("eval/contamination_report.json"))
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)

    if not args.dataset.exists():
        logging.error("Dataset not found: %s. Run doc2instruct first.", args.dataset)
        return 1

    holdout_ids, train_ids = load_manifest_splits(args.manifest)
    logging.info("Manifest: %d holdout, %d train.", len(holdout_ids), len(train_ids))
    if not holdout_ids:
        logging.error("No holdout papers in manifest; nothing to check.")
        return 1

    dataset_text, dataset_sources = dataset_text_and_sources(args.dataset)
    logging.info(
        "Dataset: %d chars of QA text, %d distinct source identifiers.",
        len(dataset_text), len(dataset_sources),
    )

    # ---- Check 1: ID leakage ----
    id_hits: list[str] = []
    for pid in holdout_ids:
        for src in dataset_sources:
            if pid in src:  # source_book may be "<id>.pdf"
                id_hits.append(pid)
                break

    # ---- Check 2: shingled text overlap ----
    holdout_texts = load_holdout_texts(
        holdout_ids, args.holdout_page_texts, args.holdout_pdf_dir
    )
    dataset_shingles = _word_shingles(dataset_text, args.shingle_size)
    logging.info("Dataset shingle universe: %d k-grams (k=%d).",
                 len(dataset_shingles), args.shingle_size)

    text_hits: list[dict] = []
    per_paper: dict[str, float] = {}
    for pid, text in holdout_texts.items():
        shingles = _word_shingles(text, args.shingle_size)
        if len(shingles) < args.min_shingles:
            per_paper[pid] = -1.0  # not enough text to judge
            continue
        overlap = len(shingles & dataset_shingles) / len(shingles)
        per_paper[pid] = round(overlap, 4)
        if overlap > args.overlap_threshold:
            text_hits.append({"paper_id": pid, "overlap": round(overlap, 4)})

    contaminated = bool(id_hits or text_hits)

    report = {
        "dataset": str(args.dataset),
        "manifest": str(args.manifest),
        "shingle_size": args.shingle_size,
        "overlap_threshold": args.overlap_threshold,
        "holdout_paper_count": len(holdout_ids),
        "id_leak_hits": sorted(set(id_hits)),
        "text_overlap_hits": sorted(text_hits, key=lambda d: -d["overlap"]),
        "per_paper_overlap": per_paper,
        "contaminated": contaminated,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    with args.report.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    logging.info("Wrote report -> %s", args.report)

    if contaminated:
        logging.error("CONTAMINATION DETECTED.")
        if id_hits:
            logging.error("  Held-out ids present in dataset metadata: %s",
                          sorted(set(id_hits)))
        if text_hits:
            logging.error("  Held-out text overlap above threshold: %s", text_hits)
        logging.error("STOP: do not fine-tune until this is resolved.")
        return 2

    logging.info("Clean: no held-out id leakage, no text overlap above %.3f.",
                 args.overlap_threshold)
    return 0


if __name__ == "__main__":
    sys.exit(main())
