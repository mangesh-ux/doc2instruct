"""Split the downloaded arXiv corpus into train and holdout sets.

Usage:
    python scripts/split_corpus.py
    python scripts/split_corpus.py --holdout-count 15 --seed 42

Output:
    corpus/train/<arxiv_id>.pdf      — symlink or copy to training PDFs
    corpus/holdout/<arxiv_id>.pdf    — symlink or copy to held-out PDFs
    corpus/manifest.jsonl            — combined manifest with `split` column

Design notes:
    - Deterministic: same seed + same source manifest -> same split forever.
    - Default: 15 papers held out (matches eval_plan.md).
    - Refuses to re-split if outputs already exist (to avoid contamination
      from accidental re-runs). Use --force to override (logs a warning).
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import shutil
import sys
from pathlib import Path

LOG_FORMAT = "%(asctime)s [%(levelname)s] %(message)s"

DEFAULT_RAW_DIR = Path("corpus/raw")
DEFAULT_TRAIN_DIR = Path("corpus/train")
DEFAULT_HOLDOUT_DIR = Path("corpus/holdout")
DEFAULT_MANIFEST_OUT = Path("corpus/manifest.jsonl")
DEFAULT_HOLDOUT_COUNT = 15
DEFAULT_SEED = 42


def load_raw_manifest(raw_dir: Path) -> list[dict]:
    manifest = raw_dir / "manifest.jsonl"
    if not manifest.exists():
        raise FileNotFoundError(
            f"Raw manifest not found at {manifest}. "
            "Run download_arxiv.py first."
        )
    records: list[dict] = []
    with manifest.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def _copy_or_link(src: Path, dst: Path) -> None:
    """Use a hardlink when possible (saves disk), fall back to copy."""
    if dst.exists():
        return
    try:
        dst.hardlink_to(src)
    except OSError:
        shutil.copy2(src, dst)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW_DIR)
    parser.add_argument("--train-dir", type=Path, default=DEFAULT_TRAIN_DIR)
    parser.add_argument("--holdout-dir", type=Path, default=DEFAULT_HOLDOUT_DIR)
    parser.add_argument("--manifest-out", type=Path, default=DEFAULT_MANIFEST_OUT)
    parser.add_argument("--holdout-count", type=int, default=DEFAULT_HOLDOUT_COUNT)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--force", action="store_true",
                        help="re-split even if output dirs are non-empty")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)

    records = load_raw_manifest(args.raw_dir)
    logging.info("Loaded %d papers from raw manifest.", len(records))

    if len(records) < args.holdout_count + 1:
        logging.error(
            "Need at least %d papers (1 train + %d holdout); got %d.",
            args.holdout_count + 1, args.holdout_count, len(records),
        )
        return 1

    # Refuse to clobber unless --force.
    for d in (args.train_dir, args.holdout_dir):
        if d.exists() and any(d.iterdir()) and not args.force:
            logging.error(
                "Output directory %s is non-empty. Refusing to re-split. "
                "Pass --force to override (this can cause contamination).",
                d,
            )
            return 1

    args.train_dir.mkdir(parents=True, exist_ok=True)
    args.holdout_dir.mkdir(parents=True, exist_ok=True)
    args.manifest_out.parent.mkdir(parents=True, exist_ok=True)

    # Deterministic shuffle + split.
    rng = random.Random(args.seed)
    shuffled = sorted(records, key=lambda r: r["arxiv_id"])  # stable order first
    rng.shuffle(shuffled)
    holdout = shuffled[: args.holdout_count]
    train = shuffled[args.holdout_count :]

    holdout_ids = {r["arxiv_id"] for r in holdout}
    logging.info("Holdout: %d papers. Train: %d papers.", len(holdout), len(train))
    logging.info("Holdout IDs: %s", sorted(holdout_ids))

    # Materialize files + write combined manifest.
    with args.manifest_out.open("w", encoding="utf-8") as out:
        for record in records:
            split = "holdout" if record["arxiv_id"] in holdout_ids else "train"
            target_dir = args.holdout_dir if split == "holdout" else args.train_dir
            src = Path(record["pdf_path"])
            dst = target_dir / src.name
            if not src.exists():
                logging.warning("Source PDF missing for %s: %s", record["arxiv_id"], src)
                continue
            _copy_or_link(src, dst)
            row = {**record, "split": split, "split_path": str(dst.as_posix())}
            out.write(json.dumps(row) + "\n")

    logging.info("Wrote combined manifest with split labels: %s", args.manifest_out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
