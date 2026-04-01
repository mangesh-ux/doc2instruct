"""Prints the exact prompts generated from the YAML config.

This helper is useful for reviewing prompt content before spending tokens.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from multimodal_dataset.config import load_config
from multimodal_dataset.pipeline import get_prompt_preview


def _resolve_book_name(config_books_dir: Path, glob_pattern: str, book_arg: str | None) -> str:
    """Resolve the book name used in prompt preview."""
    if book_arg:
        return Path(book_arg).name

    books = sorted(config_books_dir.glob(glob_pattern))
    if books:
        return books[0].name
    return "sample.pdf"


def main() -> None:
    """CLI command that prints system + user prompts."""
    parser = argparse.ArgumentParser(
        description="Show the exact system/user prompts generated from config.yaml."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config.yaml"),
        help="Path to YAML config file.",
    )
    parser.add_argument(
        "--book",
        type=str,
        default=None,
        help="Optional book filename to use in prompt preview.",
    )
    parser.add_argument(
        "--page",
        type=int,
        default=1,
        help="Page number to use in prompt preview.",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    book_name = _resolve_book_name(cfg.input.books_dir, cfg.input.glob, args.book)
    preview = get_prompt_preview(cfg, book_name=book_name, page_number=args.page)

    print("=== SYSTEM PROMPT ===")
    print(preview["system_prompt"])
    print()
    print("=== USER PROMPT ===")
    print(preview["user_prompt"])


if __name__ == "__main__":
    main()
