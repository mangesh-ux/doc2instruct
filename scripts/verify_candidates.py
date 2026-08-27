"""Programmatic verification gate for the custom eval set.

LLM-drafted QA is not trustworthy on its own: models paraphrase "verbatim"
quotes, invent specifics, and write generic questions. This script applies
*mechanical* checks that need no model, so every item that survives has an
auditable reason to be trusted.

Checks per candidate:
  1. non_empty          — question / answer / evidence all present
  2. pages_valid        — pages parse; cross_page rows have >= 2 pages
  3. evidence_grounded  — the evidence quote really occurs in the source page
                          text (normalized, hyphenation-aware, fuzzy coverage
                          >= --min-quote-coverage of the quote length)
  4. question_specific  — rejects generic templates ("main contribution", ...)
  5. answer_specific    — answer long enough to be a real fact, short enough
                          to be scoreable by EM/F1
  6. answer_not_echo    — answer is not just a copy of the question
  7. cross_page_support — for cross_page rows, distinctive answer/question terms
                          appear on BOTH pages (evidence of real multi-page
                          dependency, not a single-page question in disguise)
  8. not_duplicate      — near-duplicate questions are dropped (keeps first)

Then a SELECTION phase builds the published set from the valid pool:
  - fills explicit --target-single / --target-cross quotas,
  - round-robins across papers so no paper dominates and coverage stays even,
  - prefers items whose evidence is exactly verbatim, then richer evidence.

Separating validity from selection matters: validity is a property of an item,
selection is a property of the *set*. Mixing them (e.g. a cap applied in row
order) makes the final composition an artifact of sort order instead of a
deliberate, reportable choice.

Outputs:
  - <out-csv>                      candidates with status set to accept/reject
  - eval/custom/verification_report.json   per-check counts + rejected samples

Usage:
    python scripts/verify_candidates.py                     # verify v2 -> verified
    python scripts/verify_candidates.py --in-csv eval/custom/candidates.csv
    python scripts/verify_candidates.py --target-single 90 --target-cross 60

Design note: this gate is deliberately conservative. It is cheaper to reject a
good item than to publish an ungrounded one.
"""

from __future__ import annotations

import argparse
import csv
import difflib
import json
import logging
import re
import sys
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

LOG_FORMAT = "%(asctime)s [%(levelname)s] %(message)s"

DEFAULT_IN_CSV = Path("eval/custom/candidates_v2.csv")
DEFAULT_OUT_CSV = Path("eval/custom/candidates_verified.csv")
DEFAULT_PAGE_TEXTS = Path("eval/custom/page_texts.jsonl")
DEFAULT_REPORT = Path("eval/custom/verification_report.json")

DEFAULT_MIN_QUOTE_COVERAGE = 0.85   # fraction of the quote found verbatim
DEFAULT_MIN_QUOTE_CHARS = 25        # too-short quotes prove nothing
# 1 word is legitimate for quantitative answers ("0.85", "12"); guessable
# yes/no answers are excluded separately.
DEFAULT_MIN_ANSWER_WORDS = 1
DEFAULT_MAX_ANSWER_WORDS = 60       # keep answers EM/F1-scoreable
DEFAULT_DUP_THRESHOLD = 0.88
DEFAULT_MIN_CROSS_TERMS = 1         # distinctive terms required on each page
# Published-set composition. Single-page stays the majority (it is the bulk of
# what the pipeline produces) while cross-page comfortably exceeds the 50%-of-
# single floor required to make the Stage-2 claim measurable.
DEFAULT_TARGET_SINGLE = 90
DEFAULT_TARGET_CROSS = 60

# Question templates that don't require having read the page.
GENERIC_QUESTION_PATTERNS = [
    r"\bmain (focus|contribution|purpose|goal|idea|topic)\b",
    r"\bwhat (is|was) the (paper|study|article|work) about\b",
    r"\bwhat does (the|this) (paper|study|work) (study|investigate|explore|address)\b",
    r"\bwhat is the (purpose|aim|objective) of (the|this)\b",
    r"\bsummar(y|ize|ise)\b",
    r"\bwhat are the key (takeaways|points)\b",
]

# Words too common to prove a term came from a specific page.
_STOPWORDS = {
    "the", "and", "for", "that", "with", "this", "from", "are", "was", "were",
    "have", "has", "had", "about", "into", "their", "what", "when", "where",
    "which", "while", "would", "could", "should", "these", "those", "than",
    "then", "there", "such", "each", "both", "also", "more", "most", "some",
    "other", "between", "using", "used", "use", "can", "may", "does", "not",
    "但", "paper", "page", "passage", "author", "authors", "study", "work",
    "model", "models", "method", "methods", "result", "results", "data",
    "based", "given", "shown", "show", "shows", "same", "different", "first",
    "second", "two", "one", "how", "why", "does", "according",
}


def normalize(text: str) -> str:
    """Aggressively normalize text for robust substring matching.

    Handles the things that break naive matching on PDF-extracted text:
    unicode ligatures/quotes, hyphenated line breaks, and arbitrary whitespace.
    """
    text = unicodedata.normalize("NFKC", text)
    # Join words split across lines: "repre-\nsentation" -> "representation".
    text = re.sub(r"-\s*\n\s*", "", text)
    text = text.replace("\u2019", "'").replace("\u2018", "'")
    text = text.replace("\u201c", '"').replace("\u201d", '"')
    text = text.replace("\u2013", "-").replace("\u2014", "-")
    text = re.sub(r"\s+", " ", text)
    return text.strip().lower()


def quote_coverage(quote: str, page_text: str) -> float:
    """Fraction of the quote present as one contiguous run in the page text.

    Uses the longest common substring rather than a strict `in` test so that a
    single stray character from PDF extraction doesn't fail an otherwise
    verbatim quote. Returns 1.0 on exact containment.
    """
    q = normalize(quote)
    p = normalize(page_text)
    if not q or not p:
        return 0.0
    if q in p:
        return 1.0
    matcher = difflib.SequenceMatcher(a=q, b=p, autojunk=False)
    match = matcher.find_longest_match(0, len(q), 0, len(p))
    return match.size / len(q)


def content_terms(text: str) -> set[str]:
    """Distinctive lowercase terms (>=4 chars, non-stopword) used for overlap."""
    tokens = re.findall(r"[a-zA-Z][a-zA-Z\-]{3,}", normalize(text))
    return {t for t in tokens if t not in _STOPWORDS}


def is_generic_question(question: str) -> bool:
    q = normalize(question)
    return any(re.search(pat, q) for pat in GENERIC_QUESTION_PATTERNS)


def load_page_texts(path: Path) -> dict[tuple[str, int], str]:
    out: dict[tuple[str, int], str] = {}
    if not path.exists():
        raise FileNotFoundError(f"page_texts.jsonl not found at {path}")
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            out[(str(row["paper_id"]), int(row["page"]))] = row["text"]
    return out


def parse_pages(value: str) -> list[int]:
    return [int(p.strip()) for p in str(value).split(",") if p.strip()]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--in-csv", type=Path, nargs="+", default=[DEFAULT_IN_CSV],
                        help="one or more candidate CSVs to pool. Earlier files "
                             "win ties during selection, so list your most "
                             "trusted generation first.")
    parser.add_argument("--out-csv", type=Path, default=DEFAULT_OUT_CSV)
    parser.add_argument("--page-texts", type=Path, default=DEFAULT_PAGE_TEXTS)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--min-quote-coverage", type=float,
                        default=DEFAULT_MIN_QUOTE_COVERAGE)
    parser.add_argument("--min-quote-chars", type=int, default=DEFAULT_MIN_QUOTE_CHARS)
    parser.add_argument("--min-answer-words", type=int, default=DEFAULT_MIN_ANSWER_WORDS)
    parser.add_argument("--max-answer-words", type=int, default=DEFAULT_MAX_ANSWER_WORDS)
    parser.add_argument("--dup-threshold", type=float, default=DEFAULT_DUP_THRESHOLD)
    parser.add_argument("--target-single", type=int, default=DEFAULT_TARGET_SINGLE,
                        help="how many single_page items to publish (0 = all valid)")
    parser.add_argument("--target-cross", type=int, default=DEFAULT_TARGET_CROSS,
                        help="how many cross_page items to publish (0 = all valid)")
    parser.add_argument("--min-cross-terms", type=int, default=DEFAULT_MIN_CROSS_TERMS)
    parser.add_argument("--require-ablation-for-cross", action="store_true",
                        help="reject cross_page rows that have not been through "
                             "ablate_cross_page.py. Once the ablation exists, an "
                             "un-ablated cross-page item is unverified: 60%% of "
                             "them turned out to be single-page answerable.")
    parser.add_argument("--keep-existing-rejects", action="store_true", default=True,
                        help="rows already marked reject stay rejected")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)

    for path in args.in_csv:
        if not path.exists():
            logging.error("Input CSV not found: %s", path)
            return 1

    page_texts = load_page_texts(args.page_texts)
    rows: list[dict] = []
    fieldnames: list[str] = []
    for source_rank, path in enumerate(args.in_csv):
        with path.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for name in (reader.fieldnames or []):
                if name not in fieldnames:
                    fieldnames.append(name)
            batch = list(reader)
        for row in batch:
            row["_source_csv"] = path.name
            row["_source_rank"] = source_rank
        rows.extend(batch)
        logging.info("Loaded %d rows from %s", len(batch), path)
    logging.info("Pooled %d candidate rows from %d file(s).", len(rows), len(args.in_csv))

    for extra in ("verify_reasons", "quote_coverage", "source_csv"):
        if extra not in fieldnames:
            fieldnames.append(extra)

    reject_counter: Counter[str] = Counter()
    rejected_samples: dict[str, list[dict]] = defaultdict(list)
    accepted_signatures: list[tuple[str, str]] = []  # (normalized question, paper)
    per_paper_accepted: Counter[str] = Counter()
    coverage_values: list[float] = []

    # Deterministic order so the published set is reproducible. Source rank
    # comes last in the key so that when the same (paper, type, pages) slot is
    # present in several pooled files, the most trusted file is seen first and
    # wins the near-duplicate check.
    rows.sort(key=lambda r: (str(r.get("paper_id", "")),
                             str(r.get("question_type", "")),
                             parse_pages(r.get("pages", "0") or "0"),
                             int(r.get("_source_rank", 0))))

    for row in rows:
        reasons: list[str] = []
        paper = str(row.get("paper_id", ""))
        qtype = str(row.get("question_type", ""))
        question = (row.get("candidate_question") or "").strip()
        answer = (row.get("candidate_answer") or "").strip()
        evidence = (row.get("evidence_quote") or "").strip()
        prior_status = (row.get("status") or "").strip().lower()

        # Respect upstream rejections (e.g. the model judged the page unusable).
        if args.keep_existing_rejects and prior_status == "reject":
            row["status"] = "reject"
            row["verify_reasons"] = "upstream_reject"
            row["quote_coverage"] = ""
            reject_counter["upstream_reject"] += 1
            continue

        # 1. non_empty
        if not question or not answer or not evidence:
            reasons.append("missing_field")

        # 2. pages_valid
        try:
            pages = parse_pages(row.get("pages", ""))
        except ValueError:
            pages = []
            reasons.append("pages_unparseable")
        if not pages:
            reasons.append("pages_missing")
        if qtype == "cross_page" and len(pages) < 2:
            reasons.append("cross_page_needs_2_pages")

        # 3. evidence_grounded — the decisive check.
        # Multi-hop rows carry a second quote; each quote must be verbatim on
        # its OWN page, so one page cannot supply all the evidence.
        evidence_b = (row.get("evidence_quote_b") or "").strip()
        coverage = 0.0
        if evidence_b and len(pages) >= 2:
            per_quote = []
            for quote, page in ((evidence, pages[0]), (evidence_b, pages[1])):
                if len(normalize(quote)) < args.min_quote_chars:
                    reasons.append("evidence_too_short")
                    per_quote.append(0.0)
                    continue
                text = page_texts.get((paper, page), "")
                per_quote.append(quote_coverage(quote, text) if text else 0.0)
            # An item is only as grounded as its weakest span.
            coverage = min(per_quote) if per_quote else 0.0
            if coverage < args.min_quote_coverage:
                reasons.append(f"evidence_not_on_its_own_page(cov={coverage:.2f})")
        elif evidence and pages:
            if len(normalize(evidence)) < args.min_quote_chars:
                reasons.append("evidence_too_short")
            # Best coverage across the referenced pages.
            for p in pages:
                text = page_texts.get((paper, p), "")
                if text:
                    coverage = max(coverage, quote_coverage(evidence, text))
            if coverage < args.min_quote_coverage:
                reasons.append(f"evidence_not_in_source(cov={coverage:.2f})")
        row["quote_coverage"] = f"{coverage:.3f}"

        # 3b. multi-hop necessity, when the ablation has been run.
        # Trust the measurement, not the generator's self-report.
        verdict = (row.get("ablation_verdict") or "").strip()
        if verdict and verdict != "needs_both_pages":
            reasons.append(f"ablation:{verdict}")
        if args.require_ablation_for_cross and qtype == "cross_page" and not verdict:
            reasons.append("cross_page_not_ablation_verified")

        # 4. question_specific
        if question and is_generic_question(question):
            reasons.append("generic_question")

        # 5. answer_specific
        n_answer_words = len(answer.split())
        if answer and n_answer_words < args.min_answer_words:
            reasons.append("answer_too_short")
        if answer and n_answer_words > args.max_answer_words:
            reasons.append("answer_too_long_for_scoring")

        # 6. answer_not_echo / not guessable
        if answer and question and normalize(answer) == normalize(question):
            reasons.append("answer_echoes_question")
        # Yes/no answers are ~50% guessable, which makes them worthless for
        # measuring grounded comprehension.
        if re.fullmatch(r"(yes|no|true|false)\.?", normalize(answer)):
            reasons.append("answer_is_yes_no")

        # 7. cross_page_support — a weak lexical proxy, only needed when the
        # empirical ablation in 3b hasn't been run for this row.
        if qtype == "cross_page" and len(pages) >= 2 and not reasons and not verdict:
            probe = content_terms(f"{question} {answer}")
            per_page_hits = []
            for p in pages:
                page_term_set = content_terms(page_texts.get((paper, p), ""))
                per_page_hits.append(len(probe & page_term_set))
            if min(per_page_hits) < args.min_cross_terms:
                reasons.append("cross_page_terms_not_on_both_pages")

        # 8. not_duplicate
        if not reasons:
            q_norm = normalize(question)
            for prev_q, _prev_paper in accepted_signatures:
                if difflib.SequenceMatcher(a=q_norm, b=prev_q).ratio() >= args.dup_threshold:
                    reasons.append("near_duplicate_question")
                    break

        if reasons:
            row["status"] = "reject"
            row["verify_reasons"] = "; ".join(reasons)
            for r in reasons:
                key = r.split("(")[0]
                reject_counter[key] += 1
                if len(rejected_samples[key]) < 3:
                    rejected_samples[key].append({
                        "paper_id": paper, "pages": row.get("pages", ""),
                        "question": question[:160], "reason": r,
                    })
        else:
            # Passed validity. Selection below decides if it makes the final set.
            row["status"] = "valid"
            row["verify_reasons"] = "passed_all_checks"
            accepted_signatures.append((normalize(question), paper))

    # ---------- selection phase ----------
    # Round-robin across papers, best items first, until quotas are filled.
    valid_pool = [r for r in rows if r["status"] == "valid"]
    n_valid_single = sum(1 for r in valid_pool if r["question_type"] == "single_page")
    n_valid_cross = sum(1 for r in valid_pool if r["question_type"] == "cross_page")

    def item_rank(row: dict) -> tuple:
        """Best-first: exact verbatim evidence, trusted source, richer evidence."""
        cov = float(row.get("quote_coverage") or 0.0)
        return (-round(cov, 3), int(row.get("_source_rank", 0)),
                -len(row.get("evidence_quote") or ""))

    by_type_paper: dict[str, dict[str, list[dict]]] = {
        "single_page": defaultdict(list), "cross_page": defaultdict(list),
    }
    for row in valid_pool:
        qtype = row["question_type"]
        if qtype in by_type_paper:
            by_type_paper[qtype][str(row.get("paper_id", ""))].append(row)
    for qtype in by_type_paper:
        for paper in by_type_paper[qtype]:
            by_type_paper[qtype][paper].sort(key=item_rank)

    def select(qtype: str, target: int) -> list[dict]:
        buckets = by_type_paper.get(qtype, {})
        if not buckets:
            return []
        papers = sorted(buckets.keys())
        pool_size = sum(len(v) for v in buckets.values())
        limit = pool_size if target <= 0 else min(target, pool_size)
        chosen: list[dict] = []
        cursor = {p: 0 for p in papers}
        while len(chosen) < limit:
            progressed = False
            for p in papers:
                if len(chosen) >= limit:
                    break
                items = buckets[p]
                if cursor[p] < len(items):
                    chosen.append(items[cursor[p]])
                    cursor[p] += 1
                    progressed = True
            if not progressed:
                break
        return chosen

    selected = select("single_page", args.target_single) + \
        select("cross_page", args.target_cross)
    selected_ids = {id(r) for r in selected}

    for row in valid_pool:
        if id(row) in selected_ids:
            row["status"] = "accept"
            per_paper_accepted[str(row.get("paper_id", ""))] += 1
            coverage_values.append(float(row.get("quote_coverage") or 0.0))
        else:
            row["status"] = "reject"
            row["verify_reasons"] = "valid_but_not_selected(quota_filled)"
            reject_counter["valid_but_not_selected"] += 1

    accepted = [r for r in rows if r["status"] == "accept"]
    n_single = sum(1 for r in accepted if r["question_type"] == "single_page")
    n_cross = sum(1 for r in accepted if r["question_type"] == "cross_page")

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    for row in rows:
        row["source_csv"] = row.get("_source_csv", "")
    with args.out_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    mean_cov = sum(coverage_values) / len(coverage_values) if coverage_values else 0.0
    exact_cov = sum(1 for c in coverage_values if c >= 0.999)
    report = {
        "input_csv": [str(p) for p in args.in_csv],
        "output_csv": str(args.out_csv),
        "thresholds": {
            "min_quote_coverage": args.min_quote_coverage,
            "min_quote_chars": args.min_quote_chars,
            "min_answer_words": args.min_answer_words,
            "max_answer_words": args.max_answer_words,
            "dup_threshold": args.dup_threshold,
            "min_cross_terms": args.min_cross_terms,
        },
        "selection": {
            "target_single": args.target_single,
            "target_cross": args.target_cross,
            "valid_pool_single": n_valid_single,
            "valid_pool_cross": n_valid_cross,
        },
        "total_rows": len(rows),
        "accepted": len(accepted),
        "accepted_single_page": n_single,
        "accepted_cross_page": n_cross,
        "cross_over_single_ratio": round(n_cross / n_single, 3) if n_single else None,
        "distinct_papers_accepted": len(per_paper_accepted),
        "per_paper_accepted": dict(sorted(per_paper_accepted.items())),
        "evidence_coverage_mean_accepted": round(mean_cov, 4),
        "evidence_exact_verbatim_accepted": exact_cov,
        "accepted_by_source_csv": dict(
            Counter(r.get("_source_csv", "") for r in accepted).most_common()),
        "reject_reason_counts": dict(reject_counter.most_common()),
        "rejected_samples": {k: v for k, v in rejected_samples.items()},
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    with args.report.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    logging.info("=" * 62)
    logging.info("ACCEPTED %d / %d  (single=%d, cross=%d, cross/single=%s)",
                 len(accepted), len(rows), n_single, n_cross,
                 f"{n_cross / n_single:.2f}" if n_single else "n/a")
    logging.info("Papers represented: %d", len(per_paper_accepted))
    logging.info("Evidence coverage (accepted): mean=%.3f, exact verbatim=%d",
                 mean_cov, exact_cov)
    logging.info("Top reject reasons: %s", dict(reject_counter.most_common(8)))
    logging.info("Wrote %s and %s", args.out_csv, args.report)
    return 0


if __name__ == "__main__":
    sys.exit(main())
