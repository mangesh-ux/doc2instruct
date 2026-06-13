# CLAUDE Project Memory: doc2instruct

This file captures persistent working memory for this repository so future sessions can quickly recover full context.

## 1) Project Identity and Goal

- Project name: `doc2instruct` (repo currently structured as a multimodal PDF-to-ChatML pipeline).
- Core goal: convert long-form PDFs into grounded instruction-tuning data.
- Design principle: hierarchical two-stage generation.
  - Stage 1: single-page grounded QnA.
  - Stage 2: deterministic cross-page synthesis from evidence packs.
- Constraint focus: reliability and inspectability over novelty.
  - strict JSON schemas
  - quality gates
  - detailed telemetry
  - checkpoint-safe long runs

## 2) High-Level Architecture (Current)

- `run.py` is a thin CLI entrypoint that calls `multimodal_dataset.pipeline.main`.
- `multimodal_dataset/pipeline.py` is the orchestrator:
  - book discovery
  - page rendering
  - generation calls
  - local quality gate
  - Stage 1 artifact writing
  - Stage 2 evidence-pack synthesis
  - cross-page quality gate
  - output writing
  - logs, metrics, analytics, checkpointing, resume
- Architecture is additive: Stage 2 was added without replacing Stage 1 logic.

## 3) Stage 1 (Local / Single-Page) Memory

### Generation

- Prompt is explicitly single-page.
- Per-page generation returns structured fields including:
  - `page_status`
  - `page_status_reason`
  - `items`
- Unusable statuses include blank/unreadable handling and optional retry at higher DPI.

### Local quality gate

- Duplicate / near-duplicate filtering happens before acceptance.
- Optional model self-critique with grounding/usefulness scores.
- Citation matching check against extracted page text when available.
- Threshold-based acceptance:
  - `min_grounding_score`
  - `min_usefulness_score`
- Local decisions logged to quality JSONL.

### Local output

- Accepted items are written as ChatML records via `multimodal_dataset/chatml.py`.
- Metadata includes source and run traceability fields.

## 4) Stage 1 Intermediate Artifact (Added)

- Durable artifact path: typically `output/page_artifacts.jsonl`.
- Built per processed page in `multimodal_dataset/page_artifacts.py`.
- Includes:
  - `run_id`
  - `source_book`
  - `source_page`
  - `page_status`
  - `page_status_reason`
  - `page_text`
  - `accepted_qas`
  - lightweight signals:
    - heading candidate
    - first non-empty lines
    - normalized keywords
    - question type distribution
- No extra expensive summarization call added in v1.

## 5) Stage 2 (Cross-Page) Memory

### Evidence pack construction

- Module: `multimodal_dataset/evidence_packs.py`.
- Deterministic strategies used:
  - adjacent page windowing
  - lexical/topic overlap
  - heading continuity heuristic
- No random QA mixing as primary strategy.
- Pack contents include:
  - `source_book`
  - `source_pages`
  - `pack_id`
  - `pack_strategy`
  - `shared_terms`
  - `heading_candidates`
  - `page_texts`
  - `accepted_local_qas`
  - `evidence_snippets`

### Cross-page synthesis call

- Module: `multimodal_dataset/synthesis.py`.
- Wrapper: `synthesize_cross_page_batch(...)`.
- Takes evidence pack (not single page image).
- Strict JSON schema with:
  - `pack_status` (`usable`, `insufficient_evidence`, `redundant`, `unrelated`)
  - `pack_status_reason`
  - `items[]` with:
    - `question`
    - `answer`
    - `question_type`
    - `difficulty`
    - `requires_multi_page_reasoning`
    - `source_pages`
    - `evidence_quotes[]` (`page`, `quote`)
    - `synthesis_type` (enum category)
- Prompt grounding rule: page text is ground truth, local QAs are hints only.

### Cross-page quality gate

- Added to `multimodal_dataset/quality.py` as `critique_cross_page_item(...)`.
- Evaluates:
  - grounding
  - usefulness
  - true multi-page dependency
  - concerns list
- Returns strict JSON with:
  - `grounding_score`
  - `usefulness_score`
  - `multi_page_score`
  - `grounded`
  - `useful`
  - `truly_multi_page`
  - `concerns`
- Pipeline applies thresholds and quote consistency checks against evidence pack text.

### Cross-page outputs

- Separate output file (for cross-page accepted records), plus optional merge into main dataset.
- Cross-page quality decisions logged separately.
- Cross-page duplicates checked against both local and cross-page seen signatures.

## 6) Config Model Memory

Main loader: `multimodal_dataset/config.py` with typed dataclasses.

Key sections:

- `input`
  - PDF folder + glob
- `runtime`
  - model/runtime and resilience controls
  - retries/timeouts
  - logging paths
  - API cost estimation values
  - checkpoint settings
  - critique parallelism
- `dataset`
  - local output path
  - per-page target count
  - variety and citation preferences
- `prompts`
  - Stage 1 system prompt
- `quality`
  - Stage 1 quality gate settings
- `analytics`
  - report/token paths
- `cross_page` (added)
  - enable/disable Stage 2
  - evidence pack sizing and overlap
  - synthesis model and limits
  - artifact/output/quality paths
  - quote and threshold controls
  - merge policy into final dataset

## 7) ChatML Formatting Memory

- File: `multimodal_dataset/chatml.py`.
- Local compatibility preserved.
- Metadata extended for cross-page support:
  - `source_pages`
  - `evidence_quotes`
  - `synthesis_type`
  - `record_level` (`local` or `cross_page`)
  - `pack_id`
- Durable append writes use flush + fsync.

## 8) Logging and Telemetry Memory

The pipeline intentionally logs many artifacts:

- prompt logs (optional)
- API metrics (latency/token/cost)
- process lifecycle events
- local quality decisions
- cross-page quality decisions
- skipped pages
- failed write fallbacks
- analytics report
- token stats
- checkpoint state

Design intent: post-run debugging and auditability without rerunning full jobs.

## 9) Checkpoint and Resume Memory

Checkpoint strategy is central:

- atomic writes for checkpoint JSON
- page-level progress persistence
- resume support via `--resume`
- append-mode compatibility

State now includes both stages:

- processed pages
- counters and distributions
- seen local signatures
- cross-page completed books
- processed pack IDs
- cross-page seen signatures
- local and cross-page rejection reasons
- synthesis distributions and pack stats

Goal: allow interruption and resumption even when Stage 1 finished and Stage 2 was interrupted.

## 10) Analytics Memory

Analytics report includes classical Stage 1 metrics plus Stage 2 additions:

- local candidates/accepted
- cross-page candidates/accepted
- local rejection reasons
- cross-page rejection reasons
- synthesis type distribution
- average pages per evidence pack
- cross-page duplicate behavior
- token/cost/latency summaries

## 11) CLI and Execution Memory

Kept simple:

- `python run.py --config config.yaml`
- `--dry-run`
- `--resume`

Minimal optional flags added:

- `--skip-cross-page` (Stage 1 only)
- `--cross-page-only` (run Stage 2 using existing artifacts/checkpoint context)

## 12) Important Constraints Followed

- Stage 1 behavior preserved (not replaced).
- Stage 2 added as hierarchical extension.
- No embeddings/vector DB added in v1.
- No random QA mixing as primary cross-page strategy.
- Reused existing logging/checkpoint philosophy.
- Kept dependency footprint minimal (existing stack + stdlib heuristics).

## 13) Operational Notes / Known Realities

- Output counts are expected to be lower than raw candidate counts due to dedup and quality filters.
- API quota/rate limits can block runs; this is external to code correctness.
- `append_mode` can accumulate records across runs; use with intent.
- Cross-page-only mode assumes usable Stage 1 artifacts already exist.

## 14) Working File Map (Current Important Files)

- `run.py`
- `config.yaml`
- `config.example.yaml`
- `README.md`
- `multimodal_dataset/pipeline.py`
- `multimodal_dataset/config.py`
- `multimodal_dataset/chatml.py`
- `multimodal_dataset/openai_client.py`
- `multimodal_dataset/quality.py`
- `multimodal_dataset/pdf_pages.py`
- `multimodal_dataset/analytics.py`
- `multimodal_dataset/page_artifacts.py` (added)
- `multimodal_dataset/evidence_packs.py` (added)
- `multimodal_dataset/synthesis.py` (added)

## 15) Suggested Next Improvements (Not Yet Required)

- Factor Stage 2 orchestration into smaller helpers to reduce `pipeline.py` complexity.
- Add targeted tests for:
  - evidence pack builder determinism
  - cross-page schema and threshold behavior
  - checkpoint restore edge cases
- Add a compact run health summary command (artifact integrity checks).
- Optional: better heading extraction heuristics for scanned/structured texts.

## 16) Evaluation Pipeline Memory (added 2026-05-10)

A separate eval workstream was added to validate doc2instruct empirically.
Strategy doc: `eval_plan.md` (read this first — it locks corpus and metrics).
Operator runbook: `eval/README.md` (step-by-step commands).

Decisions locked in v1:
- Training corpus: arXiv `cs.CL`/`cs.LG`/`cs.AI`, ~150 papers from 2023-01-01.
- Held-out set: 15 papers excluded from training, used for custom eval.
- Standard benchmarks: MMLU (subset), GSM8K (Tier 1); SQuAD 2.0, Qasper, HotpotQA (Tier 2).
- Custom eval: ~150 hand-verified QA pairs (~60% single-page, ~40% cross-page).
- Models under test: Qwen 2.5 7B base, +Stage 1, +Stage 1+Stage 2.
- Anti-contamination: SHA256 + manifest tracks every paper's split label.

New scripts under `scripts/`:
- `download_arxiv.py` — pull papers via arxiv API, write `corpus/raw/manifest.jsonl`.
- `split_corpus.py` — deterministic train/holdout split (seed=42), refuses to clobber.
- `bootstrap_eval_set.py` — extract page text + emit `candidates.csv` for human review (optional `--use-openai` LLM-drafted candidates).
- `finalize_eval_set.py` — convert reviewed CSV into clean `eval/custom/test.jsonl`.
- `run_baselines.py` — wrapper around `lm-evaluation-harness` for the standard tasks.
- `run_custom_eval.py` — runs HF or OpenAI-compatible model against the custom test set; computes EM, F1, optional LLM-judge.

Extra deps: `arxiv` (for downloads). See `requirements-eval.txt`.

Order of operations: download → split → bootstrap → manual review → finalize → baselines (base model) → doc2instruct generation on training split → fine-tune (off-host, Kaggle/RunPod) → baselines + custom eval on fine-tuned model.

Open decisions (TODO before fine-tune step): LoRA rank/alpha, epoch count, judge model choice. Documented in `eval_plan.md` §7.
