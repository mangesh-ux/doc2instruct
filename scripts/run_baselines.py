"""Run lm-evaluation-harness against a base or fine-tuned model.

This is a thin wrapper around `lm-eval` that locks in the doc2instruct
benchmark suite and writes results to a predictable location.

Usage (typical, on a GPU host like Kaggle / RunPod):

    pip install lm-eval[hf,vllm]   # one-time
    python scripts/run_baselines.py \\
        --model-path Qwen/Qwen2.5-7B-Instruct \\
        --tag base_qwen25_7b

    # After fine-tuning:
    python scripts/run_baselines.py \\
        --model-path /path/to/finetuned \\
        --tag stage1_qwen25_7b

Output:
    eval/baselines/<tag>/results.json     — full lm-eval output
    eval/baselines/<tag>/summary.json     — flattened headline metrics

Tasks (matches eval_plan.md):
    Tier 1 (sanity):   mmlu, gsm8k_cot
    Tier 2 (task):     squad2, hotpotqa, qasper

Note: lm-eval task names evolve. Verify with `lm_eval --tasks list` if a task
fails to load — task IDs may have suffixed dates or been renamed.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import shutil
import subprocess
import sys
from pathlib import Path

LOG_FORMAT = "%(asctime)s [%(levelname)s] %(message)s"

# Task IDs verified against lm-evaluation-harness (v0.4.x):
#   Tier 1 (sanity):  mmlu, gsm8k_cot           -> native
#   Tier 2 (task QA):  squadv2, qasper           -> native
#                      qasper is a GROUP that runs qasper_bool + qasper_freeform
#   Tier 2 (multi-hop): hotpotqa                 -> CUSTOM task in eval/lm_eval_tasks/
# "hotpotqa" is not native to lm-eval, so we ship a real-dataset task config and
# register it via --include-path (default below). The task-availability pre-flight
# uses the same include path, and gracefully drops hotpotqa if the config or the
# dataset cannot be loaded, so the rest of the suite still runs.
DEFAULT_TASKS = ["mmlu", "gsm8k_cot", "squadv2", "qasper", "hotpotqa"]
DEFAULT_OUT_ROOT = Path("eval/baselines")
DEFAULT_INCLUDE_PATH = Path("eval/lm_eval_tasks")
DEFAULT_BATCH_SIZE = "auto"
DEFAULT_LIMIT_PER_TASK = None  # None = full task; set 200 etc. for smoke runs


def check_lm_eval_installed() -> bool:
    return shutil.which("lm_eval") is not None or shutil.which("lm-eval") is not None


def _lm_eval_bin() -> str:
    return "lm_eval" if shutil.which("lm_eval") else "lm-eval"


def filter_available_tasks(
    tasks: list[str], include_path: Path | None = None
) -> tuple[list[str], list[str]]:
    """Split requested tasks into (available, missing) using `lm_eval --tasks list`.

    Task IDs drift between harness versions, and asking lm_eval to run an unknown
    task aborts the *entire* invocation. We pre-flight the list so a single bad
    name doesn't sink the other benchmarks. Custom tasks are discovered by passing
    the same --include_path used for the run.
    """
    list_cmd = [_lm_eval_bin(), "--tasks", "list"]
    if include_path is not None:
        list_cmd.extend(["--include_path", str(include_path)])
    try:
        proc = subprocess.run(
            list_cmd,
            capture_output=True,
            text=True,
            check=False,
        )
    except Exception as exc:  # noqa: BLE001
        logging.warning("Could not list lm_eval tasks (%s); running tasks as-is.", exc)
        return tasks, []

    listing = (proc.stdout or "") + (proc.stderr or "")
    if not listing.strip():
        logging.warning("Empty task listing from lm_eval; running tasks as-is.")
        return tasks, []

    # Match whole-word task ids to avoid false positives (e.g. "squad" in "squadv2").
    available = [t for t in tasks if re.search(rf"(?<![\w-]){re.escape(t)}(?![\w-])", listing)]
    missing = [t for t in tasks if t not in available]
    return available, missing


def build_command(
    model_path: str,
    tasks: list[str],
    out_dir: Path,
    batch_size: str,
    limit: int | None,
    extra_model_args: str,
    use_vllm: bool,
    num_fewshot: int | None,
    include_path: Path | None = None,
) -> list[str]:
    bin_name = _lm_eval_bin()

    if use_vllm:
        model_arg = "vllm"
        ma = f"pretrained={model_path},dtype=bfloat16,gpu_memory_utilization=0.9"
    else:
        model_arg = "hf"
        ma = f"pretrained={model_path},dtype=bfloat16"

    if extra_model_args:
        ma = f"{ma},{extra_model_args}"

    cmd = [
        bin_name,
        "--model", model_arg,
        "--model_args", ma,
        "--tasks", ",".join(tasks),
        "--batch_size", batch_size,
        "--output_path", str(out_dir),
        "--log_samples",
    ]
    if include_path is not None:
        cmd.extend(["--include_path", str(include_path)])
    if limit is not None:
        cmd.extend(["--limit", str(limit)])
    if num_fewshot is not None:
        cmd.extend(["--num_fewshot", str(num_fewshot)])
    return cmd


def flatten_summary(results_path: Path) -> dict:
    """Extract the headline metric for each task into a flat dict."""
    if not results_path.exists():
        return {}
    with results_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    out: dict = {}
    for task, metrics in data.get("results", {}).items():
        # lm-eval returns metrics like "acc,none": 0.42, plus stderr keys
        clean = {
            k.split(",")[0]: v
            for k, v in metrics.items()
            if not k.endswith("_stderr,none") and isinstance(v, (int, float))
        }
        out[task] = clean
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True,
                        help="HF model id or local path")
    parser.add_argument("--tag", required=True,
                        help="short label used for output dir, e.g. base_qwen25_7b")
    parser.add_argument("--tasks", nargs="+", default=DEFAULT_TASKS)
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
    parser.add_argument("--include-path", type=Path, default=DEFAULT_INCLUDE_PATH,
                        help="dir of custom lm-eval task YAMLs (registers hotpotqa)")
    parser.add_argument("--batch-size", default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--limit", type=int, default=DEFAULT_LIMIT_PER_TASK,
                        help="cap items per task (smoke runs)")
    parser.add_argument("--use-vllm", action="store_true",
                        help="use vllm backend (faster, needs more VRAM)")
    parser.add_argument("--num-fewshot", type=int, default=None,
                        help="override few-shot count for all tasks")
    parser.add_argument("--model-args", default="",
                        help="extra args appended to --model_args (key=value,...)")
    parser.add_argument("--skip-task-check", action="store_true",
                        help="do not pre-flight task availability via lm_eval --tasks list")
    parser.add_argument("--dry-run", action="store_true",
                        help="print the command and exit")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)

    if not args.dry_run and not check_lm_eval_installed():
        logging.error(
            "lm_eval not found on PATH. Install with: "
            "pip install 'lm-eval[hf]' (add 'vllm' for the vllm backend)."
        )
        return 1

    # Only pass an include path that actually exists, so a missing dir doesn't
    # make lm_eval error out.
    include_path = args.include_path if args.include_path and args.include_path.exists() else None
    if args.include_path and include_path is None:
        logging.warning("--include-path %s does not exist; custom tasks (hotpotqa) "
                        "will be unavailable.", args.include_path)

    tasks = list(args.tasks)
    if not args.dry_run and not args.skip_task_check:
        available, missing = filter_available_tasks(tasks, include_path)
        if missing:
            logging.warning(
                "Skipping %d task(s) not found in this lm_eval install: %s",
                len(missing), missing,
            )
            logging.warning(
                "If you need these, register a custom task YAML and pass it via "
                "--model-args/--include-path, or update the names. See eval/README.md."
            )
        if not available:
            logging.error("None of the requested tasks are available: %s", tasks)
            return 1
        tasks = available

    out_dir = args.out_root / args.tag
    out_dir.mkdir(parents=True, exist_ok=True)

    cmd = build_command(
        model_path=args.model_path,
        tasks=tasks,
        out_dir=out_dir,
        batch_size=args.batch_size,
        limit=args.limit,
        extra_model_args=args.model_args,
        use_vllm=args.use_vllm,
        num_fewshot=args.num_fewshot,
        include_path=include_path,
    )
    logging.info("Command: %s", " ".join(cmd))

    if args.dry_run:
        return 0

    proc = subprocess.run(cmd, check=False)
    if proc.returncode != 0:
        logging.error("lm_eval exited with code %d", proc.returncode)
        return proc.returncode

    # lm-eval writes results to <out>/<model_args_safe>/results_<ts>.json — find it.
    candidates = sorted(out_dir.rglob("results*.json"))
    if not candidates:
        logging.warning("Could not locate results JSON under %s", out_dir)
        return 0
    latest = candidates[-1]
    summary = flatten_summary(latest)
    summary_path = out_dir / "summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump({"model_path": args.model_path, "tag": args.tag,
                   "results_file": str(latest), "metrics": summary}, f, indent=2)
    logging.info("Summary -> %s", summary_path)
    for task, m in summary.items():
        logging.info("  %s: %s", task, m)
    return 0


if __name__ == "__main__":
    sys.exit(main())
