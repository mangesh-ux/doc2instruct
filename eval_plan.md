# doc2instruct — Evaluation Plan

Status: locked-in v1, written 2026-05-10. Update when assumptions change.

This document fixes the *what we are evaluating* and *how we will know it worked* before any GPU work begins. Everything downstream — fine-tuning, ablations, paper writing — is driven from here.

---

## 1. Training corpus

**Decision: arXiv ML/NLP papers.**

Specifics:
- Source: arXiv API, categories `cs.CL`, `cs.LG`, `cs.AI`.
- Time window: papers from 2023-01-01 onward (recent enough to be representative, but old enough to have stable PDFs).
- Initial size: **150 papers total**, of which **15 are held out entirely from training** for the custom evaluation set.
- Each paper: tracked by arXiv ID + SHA256 hash of the PDF for contamination auditing.
- Storage layout: `corpus/raw/`, `corpus/train/`, `corpus/holdout/` with manifest JSONL files.

Why this corpus:
- Free, abundant, clean PDFs.
- Direct alignment with **Qasper** benchmark (QA over NLP papers), which gives us a free in-domain test bed.
- Strong publication framing: "instruction tuning for scientific document understanding."
- Easy to scale (initial 150 → later 1000+ if results justify it).

Out of scope for v1: textbooks, legal docs, medical docs. Defer to a domain-specialization milestone.

---

## 2. Evaluation suite (three-tier)

We evaluate at three different layers. Each tier answers a different question.

### Tier 1 — Sanity / no-regression
**Question:** Did fine-tuning damage general capability?

| Benchmark | What it measures | Notes |
|---|---|---|
| MMLU (subset) | Broad knowledge | Run a 1k-question subset for speed. |
| GSM8K | Math reasoning | Sensitive to over-fitting; useful canary. |

Run via `lm-evaluation-harness`. Goal: post-fine-tune scores within ~1–2 points of base model.

### Tier 2 — Task / document QA
**Question:** Did fine-tuning actually make the model better at document grounding?

| Benchmark | What it measures | Why included |
|---|---|---|
| SQuAD 2.0 | Single-passage extractive QA | Cheap, fast, well-known baseline. |
| Qasper | QA over NLP papers | Direct in-domain test for arXiv corpus. |
| HotpotQA | Multi-hop reasoning across docs | **Tests the Stage 2 cross-page hypothesis directly.** |

These are the headline numbers. Stage 1 ablation answered by SQuAD/Qasper; Stage 2 ablation answered by HotpotQA delta.

### Tier 3 — Custom held-out test set
**Question:** Does the model generalize on the exact distribution that doc2instruct produces?

Specs:
- **Source PDFs:** the 15 held-out arXiv papers (excluded from training).
- **Size:** 150 QA pairs (target). Hand-verified.
- **Mix:** ~60% single-page grounded QA, ~40% cross-page reasoning.
- **Format:** matches doc2instruct's ChatML output schema exactly.
- **Build process:** `bootstrap_eval_set.py` produces candidates → user reviews and fixes in a CSV → final JSONL.

This is the artifact that proves the pipeline works on its own terms. Without it, reviewers can dismiss benchmark gains as "format alignment with SQuAD."

---

## 3. Models under test

For the first round:

| Model | Why |
|---|---|
| **Qwen 2.5 7B (base)** | Apache 2.0 license, strong baseline, well-supported by Unsloth, beats Llama 3.1 8B on most benchmarks. |
| Qwen 2.5 7B + Stage 1 only fine-tune | Tests whether single-page grounded QA alone moves the needle. |
| Qwen 2.5 7B + Stage 1 + Stage 2 fine-tune | Tests the cross-page hypothesis (delta vs Stage 1 only). |

If compute allows later: add Llama 3.1 8B as a robustness check.

---

## 4. Anti-contamination protocol

Train/eval contamination is the silent killer. Rules:

1. The 15 held-out PDFs are **never** fed into doc2instruct generation.
2. Every paper is logged with `(arxiv_id, sha256, split)` in `corpus/manifest.jsonl`.
3. Before any training run, a contamination check script verifies no held-out paper's text appears in the training data (via shingled hashing).
4. Standard benchmarks (MMLU, SQuAD, etc.) are partially memorized by base models. **Only deltas vs base matter — not absolute scores.**

---

## 5. Order of operations (canonical)

```
download_arxiv.py        # pull ~150 papers
split_corpus.py          # deterministic train/holdout split
bootstrap_eval_set.py    # extract candidate QAs from holdout PDFs
[manual]                 # user reviews/refines candidates -> custom_eval.jsonl
run_baselines.py         # base Qwen 2.5 7B numbers on Tier 1 + Tier 2
[doc2instruct run]       # generate training data on training PDFs
[fine-tune]              # Unsloth + QLoRA on Kaggle / RunPod
run_baselines.py         # post-fine-tune numbers, same benchmarks
run_custom_eval.py       # post-fine-tune numbers on custom holdout
```

The fine-tuning step is the only paid/cloud-dependent step. Everything else can happen locally.

---

## 6. Success criteria for v1 milestone

We will declare v1 a success if:

- Fine-tuned model loses ≤ 2 points on Tier 1 sanity benchmarks vs base (no catastrophic forgetting).
- Fine-tuned model gains ≥ 3 points absolute on at least one of {SQuAD 2.0, Qasper}.
- Stage 1 + Stage 2 model beats Stage 1 only on HotpotQA by a non-trivial margin (≥ 1 point) — OR we have a clean negative result that we can write up honestly.
- Custom held-out F1 / LLM-judge score is meaningfully higher for fine-tuned model than base.

Negative results are fine. Unmeasurable results are not.

---

## 7. Open questions (decide before fine-tune step)

These don't block the corpus/eval prep but should be resolved before the GPU runs:

1. Exact LoRA rank/alpha for QLoRA (defaults: r=16, alpha=16). **STILL OPEN.**
2. Number of epochs (default: 1–3, watch loss curves). **STILL OPEN.**
3. ~~Whether to use HotpotQA as-is or only its multi-hop subset.~~ **RESOLVED
   (2026-06-13):** evaluate the *full real HotpotQA distractor set* for maximum
   credibility, via a custom lm-eval task at `eval/lm_eval_tasks/hotpotqa.yaml`
   (answer EM/F1, SQuAD normalization). Registered automatically by
   `run_baselines.py --include-path`.
4. ~~Whether the LLM-judge for custom eval is GPT-4o, Claude, or a strong open
   model.~~ **RESOLVED (2026-06-13):** default judge is `gpt-4.1` (strong,
   independent of the EM/F1 path). Overridable via `--judge-model`.

Mark these with TODO comments in the relevant scripts.
