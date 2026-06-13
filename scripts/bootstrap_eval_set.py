"""Bootstrap a custom held-out evaluation set from holdout PDFs.

Two modes:
  1. (default) Extract page text and emit blank candidate slots — user fills
     question/answer manually. No API calls, no costs.
  2. --use-openai  Use an LLM to draft candidate QA pairs the user can then
     accept / edit / reject. Requires OPENAI_API_KEY in .env.

Usage:
    python scripts/bootstrap_eval_set.py
    python scripts/bootstrap_eval_set.py --use-openai --per-page 2

Output:
    eval/custom/candidates.csv    — review this, fill question/answer, mark status
    eval/custom/page_texts.jsonl  — raw page text for cross-reference

After review, run finalize_eval_set.py (or manually) to convert accepted rows
into eval/custom/test.jsonl in ChatML-compatible format.

Design notes:
    - Cheap by default. Token spend only happens with --use-openai.
    - Cross-page candidates: page pairs ordered by proximity. Adjacent pairs
      (N, N+1) are preferred (strongest relation); if more are needed to hit the
      target ratio we widen the gap (N, N+2), (N, N+3), ... up to --cross-max-gap.
    - We generate at least --cross-ratio * (single-page candidates) cross-page
      candidates per paper (default 0.5), so the review pool is rich enough to
      build a genuinely multi-page eval set, not a single-page-dominated one.
    - The CSV is the primary review surface — open in a spreadsheet (or the
      review notebook at eval/review_candidates.ipynb), edit, save.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import os
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterable

try:
    import fitz  # PyMuPDF
except ImportError:  # pragma: no cover
    print("Missing dependency: pip install PyMuPDF", file=sys.stderr)
    raise

LOG_FORMAT = "%(asctime)s [%(levelname)s] %(message)s"

DEFAULT_HOLDOUT_DIR = Path("corpus/holdout")
DEFAULT_OUT_DIR = Path("eval/custom")
DEFAULT_PER_PAGE = 1            # candidate questions per page (single_page)
DEFAULT_CROSS_RATIO = 0.5       # min cross-page candidates as a fraction of single-page
DEFAULT_CROSS_MAX_GAP = 3       # widen page gap up to this to find related pairs
PAGE_TEXT_TRUNCATE = 1500       # chars in CSV (full text in page_texts.jsonl)


# ---------- candidate row schema ----------

@dataclass
class Candidate:
    paper_id: str          # short readable id (filename without ext)
    arxiv_id: str
    question_type: str     # single_page | cross_page
    pages: str             # "5" or "5,6"
    page_text_excerpt: str # first PAGE_TEXT_TRUNCATE chars, for inline review
    candidate_question: str
    candidate_answer: str
    evidence_quote: str
    status: str            # pending | accept | edit | reject
    notes: str


CSV_FIELDS = list(Candidate.__dataclass_fields__.keys())


# ---------- text extraction ----------

def extract_pages(pdf_path: Path) -> list[str]:
    """Return a list of page texts. Empty string for pages with no text."""
    pages: list[str] = []
    with fitz.open(pdf_path) as doc:
        for page in doc:
            text = page.get_text("text") or ""
            pages.append(text.strip())
    return pages


def page_is_usable(text: str, min_chars: int = 200) -> bool:
    """Filter out blank/cover/figure-only pages."""
    return len(text) >= min_chars


def cross_page_pairs(
    usable_pages: list[tuple[int, str]], max_gap: int
) -> list[tuple[tuple[int, str], tuple[int, str]]]:
    """Return candidate page pairs ordered by proximity (closest first).

    usable_pages is a list of (page_number, text) sorted by page number. We emit
    adjacent pairs first (gap=1, the strongest relation), then progressively wider
    gaps up to max_gap. Ordering by gap means a caller taking the first N pairs
    gets the most-related ones.
    """
    pairs: list[tuple[tuple[int, str], tuple[int, str]]] = []
    n = len(usable_pages)
    for gap in range(1, max_gap + 1):
        for k in range(n - gap):
            pairs.append((usable_pages[k], usable_pages[k + gap]))
    return pairs


# ---------- candidate generation ----------

def make_blank_candidate(
    paper_id: str,
    arxiv_id: str,
    pages: list[int],
    page_text: str,
    question_type: str,
) -> Candidate:
    return Candidate(
        paper_id=paper_id,
        arxiv_id=arxiv_id,
        question_type=question_type,
        pages=",".join(str(p) for p in pages),
        page_text_excerpt=page_text[:PAGE_TEXT_TRUNCATE].replace("\n", " ⏎ "),
        candidate_question="",
        candidate_answer="",
        evidence_quote="",
        status="pending",
        notes="",
    )


def make_llm_candidate(
    paper_id: str,
    arxiv_id: str,
    pages: list[int],
    page_text: str,
    question_type: str,
    client,
    model: str,
) -> Candidate | None:
    """Draft a candidate Q/A pair using the OpenAI client."""
    system = (
        "You write a single high-quality grounded reading-comprehension question "
        "from the provided page text. The answer MUST be directly supported by a "
        "short verbatim quote from the text. Avoid trivial or yes/no questions. "
        "Output strict JSON with keys: question, answer, evidence_quote."
    )
    user = (
        f"Question type: {question_type}.\n"
        f"Pages: {pages}.\n"
        f"Text:\n---\n{page_text[:6000]}\n---\n"
        "Return JSON only."
    )
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "system", "content": system},
                      {"role": "user", "content": user}],
            response_format={"type": "json_object"},
            temperature=0.2,
        )
        payload = json.loads(resp.choices[0].message.content)
        return Candidate(
            paper_id=paper_id,
            arxiv_id=arxiv_id,
            question_type=question_type,
            pages=",".join(str(p) for p in pages),
            page_text_excerpt=page_text[:PAGE_TEXT_TRUNCATE].replace("\n", " ⏎ "),
            candidate_question=payload.get("question", "").strip(),
            candidate_answer=payload.get("answer", "").strip(),
            evidence_quote=payload.get("evidence_quote", "").strip(),
            status="pending",
            notes="llm-drafted",
        )
    except Exception as exc:  # noqa: BLE001
        logging.warning("LLM candidate failed for %s pages=%s: %s",
                        paper_id, pages, exc)
        return None


# ---------- main ----------

def iter_holdout_pdfs(holdout_dir: Path) -> Iterable[Path]:
    return sorted(p for p in holdout_dir.glob("*.pdf") if p.is_file())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--holdout-dir", type=Path, default=DEFAULT_HOLDOUT_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--per-page", type=int, default=DEFAULT_PER_PAGE,
                        help="single-page candidates per usable page")
    parser.add_argument("--cross-ratio", type=float, default=DEFAULT_CROSS_RATIO,
                        help="min cross-page candidates as a fraction of single-page "
                             "candidates per paper (default 0.5 = at least 50%%)")
    parser.add_argument("--cross-max-gap", type=int, default=DEFAULT_CROSS_MAX_GAP,
                        help="widen page gap up to this value to find related pairs")
    parser.add_argument("--cross-pairs-per-paper", type=int, default=None,
                        help="hard cap on cross-page pairs per paper; overrides "
                             "--cross-ratio when set")
    parser.add_argument("--use-openai", action="store_true",
                        help="use OpenAI to pre-draft questions (costs $)")
    parser.add_argument("--openai-model", default="gpt-4o-mini",
                        help="model for --use-openai mode")
    parser.add_argument("--max-papers", type=int, default=None,
                        help="cap papers processed (debug)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    client = None
    if args.use_openai:
        try:
            from dotenv import load_dotenv
            load_dotenv()
        except ImportError:
            pass
        if not os.getenv("OPENAI_API_KEY"):
            logging.error("OPENAI_API_KEY not set; aborting --use-openai mode.")
            return 1
        from openai import OpenAI  # type: ignore
        client = OpenAI()

    pdfs = list(iter_holdout_pdfs(args.holdout_dir))
    if args.max_papers:
        pdfs = pdfs[: args.max_papers]
    if not pdfs:
        logging.error("No PDFs found in %s. Run split_corpus.py first.",
                      args.holdout_dir)
        return 1
    logging.info("Processing %d holdout PDFs.", len(pdfs))

    candidates: list[Candidate] = []
    page_texts_path = args.out_dir / "page_texts.jsonl"
    page_texts_path.unlink(missing_ok=True)

    for pdf in pdfs:
        paper_id = pdf.stem
        # arxiv_id is the same as paper_id for our naming scheme
        arxiv_id = paper_id
        try:
            pages = extract_pages(pdf)
        except Exception as exc:  # noqa: BLE001
            logging.warning("Failed to extract %s: %s", pdf, exc)
            continue

        with page_texts_path.open("a", encoding="utf-8") as f:
            for i, text in enumerate(pages, start=1):
                f.write(json.dumps({
                    "paper_id": paper_id, "page": i, "text": text,
                }) + "\n")

        usable_pages = [(i, t) for i, t in enumerate(pages, start=1) if page_is_usable(t)]
        logging.info("  %s: %d/%d pages usable", paper_id,
                     len(usable_pages), len(pages))

        # single_page candidates
        for page_num, text in usable_pages:
            for _ in range(args.per_page):
                if args.use_openai and client is not None:
                    cand = make_llm_candidate(
                        paper_id, arxiv_id, [page_num], text,
                        "single_page", client, args.openai_model,
                    )
                else:
                    cand = make_blank_candidate(
                        paper_id, arxiv_id, [page_num], text, "single_page",
                    )
                if cand is not None:
                    candidates.append(cand)

        # cross_page candidates: at least cross_ratio * (single-page count),
        # drawn from proximity-ordered page pairs (adjacent first, then wider gaps).
        n_single = len(usable_pages) * args.per_page
        if args.cross_pairs_per_paper is not None:
            cross_target = args.cross_pairs_per_paper
        else:
            cross_target = math.ceil(args.cross_ratio * n_single)
        all_pairs = cross_page_pairs(usable_pages, args.cross_max_gap)
        selected_pairs = all_pairs[:cross_target]
        logging.info(
            "  %s: %d single-page, targeting %d cross-page (%d pairs available)",
            paper_id, n_single, len(selected_pairs), len(all_pairs),
        )
        for (p_a, t_a), (p_b, t_b) in selected_pairs:
            combined = f"--- page {p_a} ---\n{t_a}\n\n--- page {p_b} ---\n{t_b}"
            if args.use_openai and client is not None:
                cand = make_llm_candidate(
                    paper_id, arxiv_id, [p_a, p_b], combined,
                    "cross_page", client, args.openai_model,
                )
            else:
                cand = make_blank_candidate(
                    paper_id, arxiv_id, [p_a, p_b], combined, "cross_page",
                )
            if cand is not None:
                candidates.append(cand)

    # Write CSV for human review.
    csv_path = args.out_dir / "candidates.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for cand in candidates:
            writer.writerow(asdict(cand))
    logging.info("Wrote %d candidates -> %s", len(candidates), csv_path)
    logging.info("Page texts -> %s", page_texts_path)
    logging.info(
        "Next step: open candidates.csv in a spreadsheet, fill in questions/"
        "answers, mark status=accept on rows you keep. Then run "
        "finalize_eval_set.py (manual conversion: see eval/README.md)."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
