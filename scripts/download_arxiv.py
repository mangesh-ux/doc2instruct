"""Download arXiv ML/NLP papers as the doc2instruct training corpus.

Usage:
    python scripts/download_arxiv.py --limit 150
    python scripts/download_arxiv.py --limit 5 --output-dir corpus/raw_smoke   # smoke test

Output:
    corpus/raw/<arxiv_id>.pdf       — downloaded PDFs
    corpus/raw/manifest.jsonl       — one record per paper with metadata + SHA256

Design notes:
    - Uses the official `arxiv` Python client (rate-limited, polite).
    - Deterministic: papers ordered by arXiv `submittedDate` descending; given the
      same query and limit, re-runs hit the same set.
    - Idempotent: skips papers already on disk (matched by arxiv_id), so partial
      runs can resume by re-invoking with the same limit.
    - Manifest is append-mode JSONL — survives crashes mid-batch.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
import time
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Iterator

try:
    import arxiv
except ImportError:  # pragma: no cover
    print(
        "Missing dependency: install with `pip install arxiv tqdm`",
        file=sys.stderr,
    )
    raise

import requests  # transitive dep of arxiv; used for direct PDF download

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover
    tqdm = lambda x, **kwargs: x  # type: ignore  # noqa: E731


# ---------- config ----------

DEFAULT_CATEGORIES = ["cs.CL", "cs.LG", "cs.AI"]
DEFAULT_OUTPUT_DIR = Path("corpus/raw")
DEFAULT_LIMIT = 150
DEFAULT_START_DATE = "2023-01-01"
DEFAULT_SLEEP_SECONDS = 3.0  # arXiv asks for politeness; their rate-limit guidance is ~3s
PDF_DOWNLOAD_TIMEOUT = 60
# Polite identifying UA per arXiv API usage guidance.
HTTP_HEADERS = {"User-Agent": "doc2instruct-eval/1.0 (https://arxiv.org; research use)"}

LOG_FORMAT = "%(asctime)s [%(levelname)s] %(message)s"


@dataclass
class PaperRecord:
    """One row in manifest.jsonl."""
    arxiv_id: str
    title: str
    authors: list[str]
    primary_category: str
    categories: list[str]
    published: str  # ISO 8601 date
    pdf_path: str
    sha256: str
    size_bytes: int
    download_time: str


# ---------- helpers ----------

def _sha256_of_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _build_query(categories: Iterable[str], start_date: str) -> str:
    """Build an arXiv API query string.

    arXiv's API uses `cat:` for category and supports boolean OR.
    Date filtering is done by `submittedDate:[start TO now]`.
    """
    cat_clause = " OR ".join(f"cat:{c}" for c in categories)
    # Format dates as YYYYMMDDHHMM as arXiv expects.
    start = start_date.replace("-", "") + "0000"
    end = datetime.now(timezone.utc).strftime("%Y%m%d%H%M")
    return f"({cat_clause}) AND submittedDate:[{start} TO {end}]"


def _existing_ids(output_dir: Path) -> set[str]:
    """Return arxiv_ids already in the manifest, for resume support."""
    manifest = output_dir / "manifest.jsonl"
    if not manifest.exists():
        return set()
    out: set[str] = set()
    with manifest.open("r", encoding="utf-8") as f:
        for line in f:
            try:
                out.add(json.loads(line)["arxiv_id"])
            except (json.JSONDecodeError, KeyError):
                continue
    return out


def _normalize_arxiv_id(entry_id: str) -> str:
    """Convert 'http://arxiv.org/abs/2401.12345v2' -> '2401.12345'."""
    tail = entry_id.rsplit("/", 1)[-1]
    return tail.split("v")[0]


# ---------- core ----------

def search_papers(
    categories: list[str],
    start_date: str,
    limit: int,
    page_size: int = 100,
) -> Iterator[arxiv.Result]:
    """Yield arXiv search results lazily."""
    query = _build_query(categories, start_date)
    logging.info("arXiv query: %s", query)

    client = arxiv.Client(
        page_size=page_size,
        delay_seconds=DEFAULT_SLEEP_SECONDS,
        num_retries=3,
    )
    search = arxiv.Search(
        query=query,
        max_results=limit,
        sort_by=arxiv.SortCriterion.SubmittedDate,
        sort_order=arxiv.SortOrder.Descending,
    )
    yield from client.results(search)


def download_paper(result: arxiv.Result, output_dir: Path) -> PaperRecord | None:
    """Download a single paper. Returns None if the download failed.

    The `arxiv` client dropped `Result.download_pdf` in 4.x, so we fetch the PDF
    directly from `result.pdf_url`. We verify the payload is actually a PDF (the
    server occasionally returns an HTML error/"not ready" page) before keeping it.
    """
    arxiv_id = _normalize_arxiv_id(result.entry_id)
    pdf_path = output_dir / f"{arxiv_id}.pdf"

    pdf_url = getattr(result, "pdf_url", None)
    if not pdf_url:
        logging.warning("No pdf_url available for %s", arxiv_id)
        return None

    try:
        resp = requests.get(
            pdf_url, headers=HTTP_HEADERS, timeout=PDF_DOWNLOAD_TIMEOUT, allow_redirects=True
        )
        resp.raise_for_status()
        content = resp.content
        if not content[:5].startswith(b"%PDF"):
            logging.warning(
                "Response for %s was not a PDF (got %d bytes, content-type=%s); skipping.",
                arxiv_id, len(content), resp.headers.get("Content-Type", "?"),
            )
            return None
        pdf_path.write_bytes(content)
    except Exception as exc:  # noqa: BLE001
        logging.warning("Failed to download %s: %s", arxiv_id, exc)
        return None

    if not pdf_path.exists() or pdf_path.stat().st_size == 0:
        logging.warning("Downloaded file missing/empty for %s", arxiv_id)
        return None

    sha = _sha256_of_file(pdf_path)
    return PaperRecord(
        arxiv_id=arxiv_id,
        title=result.title.strip(),
        authors=[a.name for a in result.authors],
        primary_category=result.primary_category,
        categories=list(result.categories),
        published=result.published.isoformat(),
        pdf_path=str(pdf_path.as_posix()),
        sha256=sha,
        size_bytes=pdf_path.stat().st_size,
        download_time=datetime.now(timezone.utc).isoformat(),
    )


def append_manifest(record: PaperRecord, output_dir: Path) -> None:
    manifest = output_dir / "manifest.jsonl"
    with manifest.open("a", encoding="utf-8") as f:
        f.write(json.dumps(asdict(record)) + "\n")
        f.flush()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=DEFAULT_LIMIT,
                        help=f"max papers to download (default: {DEFAULT_LIMIT})")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR,
                        help=f"where to save PDFs (default: {DEFAULT_OUTPUT_DIR})")
    parser.add_argument("--categories", nargs="+", default=DEFAULT_CATEGORIES,
                        help=f"arXiv categories (default: {DEFAULT_CATEGORIES})")
    parser.add_argument("--start-date", default=DEFAULT_START_DATE,
                        help=f"earliest submitted date (YYYY-MM-DD, default: {DEFAULT_START_DATE})")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format=LOG_FORMAT,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    seen = _existing_ids(args.output_dir)
    logging.info("Resume: %d papers already in manifest, skipping those.", len(seen))

    target_new = max(0, args.limit - len(seen))
    if target_new == 0:
        logging.info("Already at limit (%d). Nothing to do.", args.limit)
        return 0

    # Over-fetch a bit so skipped duplicates don't shrink the final set.
    fetch_budget = target_new + len(seen) + 20
    logging.info("Fetching up to %d candidates to land %d new papers.",
                 fetch_budget, target_new)

    downloaded = 0
    skipped = 0
    failed = 0

    results = search_papers(args.categories, args.start_date, fetch_budget)
    pbar = tqdm(total=target_new, desc="Downloading", unit="paper")

    for result in results:
        if downloaded >= target_new:
            break
        arxiv_id = _normalize_arxiv_id(result.entry_id)
        if arxiv_id in seen:
            skipped += 1
            continue

        record = download_paper(result, args.output_dir)
        if record is None:
            failed += 1
            continue

        append_manifest(record, args.output_dir)
        seen.add(arxiv_id)
        downloaded += 1
        pbar.update(1)
        # arXiv client already throttles per-page; we're only sleeping between
        # downloads as light extra politeness.
        time.sleep(0.5)

    pbar.close()
    logging.info(
        "Done. downloaded=%d, skipped_existing=%d, failed=%d, manifest=%s",
        downloaded, skipped, failed,
        args.output_dir / "manifest.jsonl",
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
