"""Helpers for writing run-level analytics artifacts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def write_analytics_report(report_path: Path, report: dict[str, Any]) -> None:
    """Write formatted JSON analytics report to disk."""
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
