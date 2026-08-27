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

1. ~~Exact LoRA rank/alpha for QLoRA.~~ **RESOLVED (2026-08-27):** r=16,
   alpha=32, dropout 0.0, all attention + MLP projections. alpha = 2r is the
   stable default and lets a small adapter still move the model on a
   few-thousand-example set; including the MLP projections matters because this
   dataset teaches factual recall. Set in `eval/finetune_qlora.ipynb`.
2. ~~Number of epochs.~~ **RESOLVED (2026-08-27):** 2 epochs, effective batch
   16, lr 2e-4 cosine. At this dataset size 1 epoch underfits and 3+ starts
   memorising answers verbatim, which would inflate our own custom eval. The
   notebook prints the loss curve; drop to 1 if train loss falls below ~0.5
   early.
3. ~~Whether to use HotpotQA as-is or only its multi-hop subset.~~ **RESOLVED
   (2026-06-13):** evaluate the *full real HotpotQA distractor set* for maximum
   credibility, via a custom lm-eval task at `eval/lm_eval_tasks/hotpotqa.yaml`
   (answer EM/F1, SQuAD normalization). Registered automatically by
   `run_baselines.py --include-path`.
4. ~~Whether the LLM-judge for custom eval is GPT-4o, Claude, or a strong open
   model.~~ **RESOLVED (2026-06-13):** default judge is `gpt-4.1` (strong,
   independent of the EM/F1 path). Overridable via `--judge-model`.

Mark these with TODO comments in the relevant scripts.

---

## 8. Validation findings (2026-08-27)

Before committing to the full-corpus generation run, the pipeline was smoke-run
on 2 arXiv papers × 5 pages (`config.smoke.yaml`, outputs in
`output/smoke_run/`). Every prior run had been on a scanned book, so this was
the first time the pipeline saw born-digital papers. Three defects surfaced that
would each have quietly damaged the results.

### 8.1 The citation matcher rejected almost everything (fixed)

`has_citation_match` normalized only case and whitespace, then required an exact
substring. The generator transcribes quotes from the page *image* while the check
runs against *PyMuPDF-extracted* text; the two differ on ligatures, hyphenated
line breaks and maths spacing even when the quote is faithful.

Worse, `_cross_quote_matches` compared with a bare `.lower()`, leaving PDF
newlines in the page text while model quotes are single-line — so cross-page
matching could never succeed. **Stage 2 produced exactly zero records.**

A failed citation also clamps `grounding_score` to 0.2, which then trips
`grounding_score_below_threshold`, so one brittle matcher manufactured two
rejection reasons and looked like a model-quality problem.

Fix: a shared coverage-based matcher (`quality.normalize_for_match`,
`quote_coverage`, `citation_coverage`) that folds unicode, rejoins hyphenated
line breaks, and accepts ≥0.85 contiguous coverage.

### 8.2 Models elide quotes with "..." (fixed)

Remaining failures clustered at coverage 0.5–0.7 because the model joined
non-adjacent sentences with an ellipsis. Each fragment was verbatim, but no
single contiguous span matched. Fix: `citation_coverage` splits on ellipses and
requires *every* fragment to be found; the generation prompt now also asks for
one contiguous span. Verified on the real failures: 9 of 16 recovered, all of
them elided quotes, while single-span paraphrases at coverage 0.36–0.73
correctly stayed rejected.

**Net effect on a fixed 40-candidate sample:**

| | Before | After |
|---|---|---|
| Local records accepted | 9 | 25 |
| Cross-page records accepted | 0 | 9 |
| Total records | 9 | 34 |

### 8.3 Outputs were about to be mixed with an unrelated run (fixed)

`config.train.yaml` wrote to `./output/` with `append_mode: true`, and
`output/chatml_dataset.jsonl` already held 53 records from a scanned book. The
published training set would have silently contained them. All paths are now
namespaced under `output/train_run/`.

### 8.4 Measured cost and time for the full run

From the validated smoke run: 10 pages → 34 records, 66 API calls, 97,765 input
and 16,766 output tokens, 155 s wall clock.

Extrapolated to the 135-paper training split (2,686 pages):

| | Estimate |
|---|---|
| Records | ~9,100 |
| Wall clock | ~11.6 h (pages are processed sequentially; only critique is parallel) |
| Cost at config's assumed $5/$15 per 1M | ~$199 |
| Cost at gpt-4.1 list price ($2/$8 per 1M) | ~$89 |

The config's cost constants are deliberately conservative, so treat ~$90 as the
expected spend and ~$199 as the ceiling. Time, not money, is the binding
constraint: `parallel_critique_workers` only parallelises the judge, so
page-level generation is serial. Use `--resume` for interruptions.

### 8.5 "Multi-hop" cross-page questions mostly were not (fixed)

The first cross-page eval subset was invalid: 57 of 58 items had all their
evidence on one page, because the candidate schema has a single
`evidence_quote` column and cannot express a two-page item.

Rebuilding with one verbatim span *per page* was not sufficient either — the
generator self-certified items whose answer sat entirely inside one page's span.
`scripts/ablate_cross_page.py` now tests it: answer using only page A, then only
page B, and keep the item only if neither reaches token F1 ≥ 0.6 against the
gold answer.

**Of 136 self-certified multi-hop items, 82 (60.3%) were single-page
answerable.** 54 survived the ablation; 35 also passed per-page verbatim
grounding and form the published `cross_page` subset.

Consequence for §2: any cross-page result must be read alongside HotpotQA, and
the cross-page subset must be reported separately from single-page.
