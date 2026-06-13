"""Run a model against the custom held-out eval set and score it.

Two backends:
    --backend hf     local HuggingFace model (needs a GPU)
    --backend openai any OpenAI-compatible endpoint (api.openai.com,
                     local vllm OpenAI server, RunPod inference, etc.)

Metrics:
    - Exact Match (EM, normalized).
    - SQuAD-style token F1.
    - Optional LLM-as-judge correctness score (--judge).

Usage:
    # Base model on local GPU
    python scripts/run_custom_eval.py \\
        --backend hf \\
        --model-path Qwen/Qwen2.5-7B-Instruct \\
        --tag base_qwen25_7b

    # Fine-tuned model
    python scripts/run_custom_eval.py \\
        --backend hf \\
        --model-path /path/to/finetuned \\
        --tag stage1_qwen25_7b

Output:
    eval/custom/results/<tag>/predictions.jsonl
    eval/custom/results/<tag>/scores.json
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import string
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Iterable

LOG_FORMAT = "%(asctime)s [%(levelname)s] %(message)s"

DEFAULT_TEST_PATH = Path("eval/custom/test.jsonl")
DEFAULT_PAGE_TEXTS = Path("eval/custom/page_texts.jsonl")
DEFAULT_OUT_ROOT = Path("eval/custom/results")
DEFAULT_MAX_NEW_TOKENS = 256
DEFAULT_CONTEXT_CHAR_LIMIT = 8000  # truncate context fed to the model


# ---------- metric helpers (SQuAD-style) ----------

def _normalize(text: str) -> str:
    text = text.lower()
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    text = "".join(ch for ch in text if ch not in set(string.punctuation))
    text = re.sub(r"\s+", " ", text).strip()
    return text


def exact_match(pred: str, gold: str) -> int:
    return int(_normalize(pred) == _normalize(gold))


def f1_score(pred: str, gold: str) -> float:
    p_tokens = _normalize(pred).split()
    g_tokens = _normalize(gold).split()
    if not p_tokens or not g_tokens:
        return float(p_tokens == g_tokens)
    common = Counter(p_tokens) & Counter(g_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(p_tokens)
    recall = num_same / len(g_tokens)
    return 2 * precision * recall / (precision + recall)


# ---------- context loading ----------

def load_page_texts(path: Path) -> dict[tuple[str, int], str]:
    """Map (paper_id, page_number) -> page text."""
    out: dict[tuple[str, int], str] = {}
    if not path.exists():
        logging.warning("Page texts not found at %s — context will be empty.", path)
        return out
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            out[(row["paper_id"], int(row["page"]))] = row["text"]
    return out


def build_context(item: dict, page_texts: dict[tuple[str, int], str],
                  char_limit: int) -> str:
    parts = []
    for p in item["pages"]:
        text = page_texts.get((item["paper_id"], int(p)), "")
        parts.append(f"--- page {p} ---\n{text}")
    ctx = "\n\n".join(parts)
    if len(ctx) > char_limit:
        ctx = ctx[:char_limit] + "\n[...truncated]"
    return ctx


def build_messages(item: dict, context: str) -> list[dict]:
    system = (
        "You answer reading-comprehension questions about scientific papers. "
        "Answer concisely using only information from the provided context. "
        "If the context is insufficient, say 'unanswerable'."
    )
    user = (
        f"Context:\n{context}\n\n"
        f"Question: {item['question']}\n\n"
        "Answer concisely:"
    )
    return [{"role": "system", "content": system},
            {"role": "user", "content": user}]


# ---------- backends ----------

class HFBackend:
    def __init__(self, model_path: str, max_new_tokens: int):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        import torch
        self.tok = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
        self.max_new_tokens = max_new_tokens

    def generate(self, messages: list[dict]) -> str:
        inputs = self.tok.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
        ).to(self.model.device)
        prompt_len = inputs["input_ids"].shape[1]
        out = self.model.generate(
            **inputs,
            max_new_tokens=self.max_new_tokens,
            do_sample=False,
            pad_token_id=self.tok.pad_token_id or self.tok.eos_token_id,
        )
        # Decode ONLY the newly generated tokens. Decoding the whole sequence and
        # string-stripping the prompt is unreliable because special tokens get
        # dropped by skip_special_tokens, so the prefix never matches and the
        # entire prompt leaks into the "prediction".
        gen_tokens = out[0][prompt_len:]
        return self.tok.decode(gen_tokens, skip_special_tokens=True).strip()


class OpenAIBackend:
    def __init__(self, model: str, base_url: str | None, max_new_tokens: int):
        from openai import OpenAI
        kwargs = {}
        if base_url:
            kwargs["base_url"] = base_url
        self.client = OpenAI(**kwargs)
        self.model = model
        self.max_new_tokens = max_new_tokens

    def generate(self, messages: list[dict]) -> str:
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=self.max_new_tokens,
            temperature=0.0,
        )
        return resp.choices[0].message.content or ""


# ---------- judge (optional) ----------

def llm_judge(client_backend, item: dict, prediction: str) -> dict:
    """Cheap correctness judge. Returns {'correct': bool, 'reason': str}."""
    msg = [
        {"role": "system",
         "content": "You judge whether a predicted answer is semantically "
                    "correct given the reference answer. Output JSON: "
                    "{\"correct\": true|false, \"reason\": str}."},
        {"role": "user",
         "content": (
             f"Question: {item['question']}\n"
             f"Reference answer: {item['answer']}\n"
             f"Predicted answer: {prediction}\n"
             "JSON only."
         )},
    ]
    raw = client_backend.generate(msg)
    try:
        # tolerate stray prose
        j_start = raw.find("{")
        j_end = raw.rfind("}")
        if j_start == -1 or j_end == -1:
            return {"correct": False, "reason": f"unparseable: {raw[:80]}"}
        return json.loads(raw[j_start:j_end + 1])
    except Exception as exc:  # noqa: BLE001
        return {"correct": False, "reason": f"parse_error: {exc}"}


# ---------- main ----------

def load_test_set(path: Path) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(f"Test set not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test-path", type=Path, default=DEFAULT_TEST_PATH)
    parser.add_argument("--page-texts", type=Path, default=DEFAULT_PAGE_TEXTS)
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--backend", choices=["hf", "openai"], default="hf")
    parser.add_argument("--model-path", required=True,
                        help="HF id/path or OpenAI model name")
    parser.add_argument("--openai-base-url", default=None,
                        help="for vllm/runpod OpenAI-compatible servers")
    parser.add_argument("--max-new-tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS)
    parser.add_argument("--context-char-limit", type=int,
                        default=DEFAULT_CONTEXT_CHAR_LIMIT)
    parser.add_argument("--judge", action="store_true",
                        help="run an LLM-as-judge pass for soft correctness")
    parser.add_argument("--judge-model", default="gpt-4o-mini")
    parser.add_argument("--limit", type=int, default=None,
                        help="cap items (smoke run)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)

    items = load_test_set(args.test_path)
    if args.limit:
        items = items[: args.limit]
    page_texts = load_page_texts(args.page_texts)
    logging.info("Loaded %d items, %d cached pages", len(items), len(page_texts))

    if args.backend == "hf":
        backend = HFBackend(args.model_path, args.max_new_tokens)
    else:
        backend = OpenAIBackend(args.model_path, args.openai_base_url,
                                args.max_new_tokens)

    judge_backend = None
    if args.judge:
        # Build the judge once (always OpenAI), not once per item.
        judge_backend = OpenAIBackend(args.judge_model, None, max_new_tokens=200)

    out_dir = args.out_root / args.tag
    out_dir.mkdir(parents=True, exist_ok=True)
    pred_path = out_dir / "predictions.jsonl"

    em_total = 0.0
    f1_total = 0.0
    n_single = 0
    n_cross = 0
    em_single = 0.0
    em_cross = 0.0
    f1_single = 0.0
    f1_cross = 0.0
    judge_correct = 0
    t0 = time.time()

    with pred_path.open("w", encoding="utf-8") as fout:
        for i, item in enumerate(items, start=1):
            ctx = build_context(item, page_texts, args.context_char_limit)
            messages = build_messages(item, ctx)
            try:
                pred = backend.generate(messages)
            except Exception as exc:  # noqa: BLE001
                logging.warning("Generation failed for %s: %s", item["id"], exc)
                pred = ""

            em = exact_match(pred, item["answer"])
            f1 = f1_score(pred, item["answer"])
            em_total += em
            f1_total += f1

            judge_result = None
            if judge_backend is not None:
                judge_result = llm_judge(judge_backend, item, pred)
                if judge_result.get("correct"):
                    judge_correct += 1

            if item["question_type"] == "single_page":
                n_single += 1
                em_single += em
                f1_single += f1
            else:
                n_cross += 1
                em_cross += em
                f1_cross += f1

            fout.write(json.dumps({
                "id": item["id"],
                "question_type": item["question_type"],
                "question": item["question"],
                "gold": item["answer"],
                "prediction": pred,
                "em": em, "f1": f1,
                "judge": judge_result,
            }) + "\n")

            if i % 10 == 0:
                logging.info("  [%d/%d] running EM=%.3f F1=%.3f",
                             i, len(items), em_total / i, f1_total / i)

    n = len(items) or 1
    summary = {
        "tag": args.tag,
        "model_path": args.model_path,
        "n_items": n,
        "em": em_total / n,
        "f1": f1_total / n,
        "single_page": {
            "n": n_single,
            "em": (em_single / n_single) if n_single else None,
            "f1": (f1_single / n_single) if n_single else None,
        },
        "cross_page": {
            "n": n_cross,
            "em": (em_cross / n_cross) if n_cross else None,
            "f1": (f1_cross / n_cross) if n_cross else None,
        },
        "judge_correct_rate": (judge_correct / n) if args.judge else None,
        "elapsed_seconds": time.time() - t0,
    }
    with (out_dir / "scores.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    logging.info("=== Final ===")
    logging.info(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
