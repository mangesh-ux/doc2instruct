# Multimodal PDF to ChatML Dataset Pipeline

doc2instruct turns long-form PDFs into grounded instruction-tuning data through a hierarchical two-stage pipeline. It first generates high-quality QnA from individual pages, then builds deterministic evidence packs across related pages to synthesize multi-page supervision that a single-page prompt cannot capture. The result is a more reliable path from raw documents to ChatML datasets: explicit grounding, structured quality gates, detailed telemetry, and checkpoint-safe long runs.

## Overview

This project converts PDF books into ChatML training data with two stages:

1. **Stage 1 (pagewise):** Generate grounded QnA from single pages.
2. **Stage 2 (cross-page):** Build deterministic evidence packs and synthesize multi-page QnA.

The pipeline is designed for reliability: strict JSON outputs, quality gates, detailed logs, and checkpoint/resume support.

## Outputs

- `output/chatml_dataset.jsonl`: main dataset output
- `output/cross_page_chatml_dataset.jsonl`: cross-page records only
- `output/page_artifacts.jsonl`: Stage 1 page artifacts used by Stage 2
- `output/skipped_pages.jsonl`: unusable/blank page log
- `output/quality_log.jsonl`: Stage 1 quality decisions
- `output/cross_page_quality_log.jsonl`: Stage 2 quality decisions
- `output/prompt_log.jsonl`: prompt trace (optional)
- `output/api_metrics.jsonl`: API latency/token/cost telemetry
- `output/process_log.jsonl`: process lifecycle events
- `output/analytics_report.json`: run analytics
- `output/token_stats.json`: token summary by call type
- `output/run_checkpoint.json`: checkpoint state for resume
- `output/failed_writes.jsonl`: fallback write-failure log

## How It Works

### Stage 1: Local grounding

- Render each PDF page to an image data URL.
- Generate QnA from that single page.
- Retry blank/unreadable pages at higher DPI when configured.
- Run quality filters:
  - duplicate and near-duplicate rejection
  - model critique (optional)
  - citation checks against extracted page text
- Write accepted local records to ChatML.
- Write page artifacts for Stage 2.

### Stage 2: Cross-page synthesis

- Build deterministic evidence packs from Stage 1 artifacts using:
  - adjacent page windowing
  - lexical overlap
  - heading continuity
- Synthesize cross-page QnA from evidence packs.
- Require evidence-based multi-page reasoning in schema.
- Run cross-page quality checks for:
  - grounding
  - usefulness
  - true multi-page dependency
  - quote/page consistency
  - duplicate similarity
- Write cross-page results separately, and optionally merge into main output.

## Quick Start

1) Create and activate a virtual environment.

2) Install dependencies:

```bash
pip install -r requirements.txt
```

3) Create env file:

```bash
copy .env.example .env
```

Set your key in `.env`:

```env
OPENAI_API_KEY=...
```

4) Create config:

```bash
copy config.example.yaml config.yaml
```

5) Put PDF files in `books/`.

6) Run a cheap smoke test:

```bash
python run.py --config config.yaml --dry-run
```

7) Optional prompt preview:

```bash
python show_prompts.py --config config.yaml --page 1
```

8) Run full pipeline:

```bash
python run.py --config config.yaml
```

9) Resume from checkpoint:

```bash
python run.py --config config.yaml --resume
```

10) Optional mode flags:

```bash
python run.py --config config.yaml --skip-cross-page
python run.py --config config.yaml --cross-page-only --resume
```

## Fast Validation Checklist

After a test run, verify:

- `output/chatml_dataset.jsonl` exists
- `output/analytics_report.json` exists
- `output/process_log.jsonl` contains `run_started` and `run_completed`
- if cross-page is enabled, `output/page_artifacts.jsonl` exists

## Configuration Areas

### Runtime

- `runtime.model`
- `runtime.max_pages_per_book`
- `runtime.dpi`
- `runtime.retry_dpi`
- `runtime.max_unusable_retries`
- `runtime.request_timeout_seconds`
- `runtime.parallel_critique_workers`
- `runtime.parallel_future_timeout_seconds`

### Stage 1 dataset

- `dataset.output_path`
- `dataset.qas_per_page`
- `dataset.user_profile`
- `dataset.variety`
- `dataset.citation`

### Stage 1 quality

- `quality.enabled`
- `quality.use_model_self_critique`
- `quality.critique_model`
- `quality.min_grounding_score`
- `quality.min_usefulness_score`
- `quality.duplicate_similarity_threshold`
- `quality.require_citation_match_if_text_available`

### Stage 2 cross-page

- `cross_page.enabled`
- `cross_page.min_pages_per_pack`
- `cross_page.max_pages_per_pack`
- `cross_page.pack_overlap_window`
- `cross_page.max_cross_page_qas_per_pack`
- `cross_page.max_evidence_quotes_per_item`
- `cross_page.use_local_qas_as_hints`
- `cross_page.synthesis_model`
- `cross_page.synthesis_temperature`
- `cross_page.output_path`
- `cross_page.artifact_path`
- `cross_page.quality_log_path`
- `cross_page.require_quote_match_if_text_available`
- `cross_page.min_cross_page_grounding_score`
- `cross_page.min_cross_page_usefulness_score`
- `cross_page.min_multi_page_score`
- `cross_page.merge_into_final_dataset`

## Output Interpretation

- `local_candidates`: raw Stage 1 generated items before final filtering
- `local_accepted`: Stage 1 records that passed quality gate
- `cross_page_candidates`: raw Stage 2 generated items before final filtering
- `cross_page_accepted`: Stage 2 records that passed cross-page quality gate
- final dataset count is usually lower than candidate counts due to filtering

## Project Structure

- `run.py`: CLI entrypoint
- `show_prompts.py`: prompt preview utility
- `multimodal_dataset/pipeline.py`: main orchestration
- `multimodal_dataset/config.py`: typed config loader
- `multimodal_dataset/openai_client.py`: generation API wrapper
- `multimodal_dataset/quality.py`: quality logic and critique wrappers
- `multimodal_dataset/pdf_pages.py`: rendering and text extraction
- `multimodal_dataset/chatml.py`: record formatting and JSONL append
- `multimodal_dataset/analytics.py`: analytics writer
- `multimodal_dataset/page_artifacts.py`: Stage 1 page artifact builder
- `multimodal_dataset/evidence_packs.py`: deterministic evidence pack builder
- `multimodal_dataset/synthesis.py`: Stage 2 synthesis wrapper/schema

## Notes

- Use `--dry-run` before longer runs.
- Keep `.env` private and never commit API keys.
- PDF files in `books/` are ignored by git.
- Cost metrics are estimates based on configured token prices.
- `run_id` is attached to accepted records for traceability.
- Checkpoint/resume supports long-running jobs.
# Multimodal PDF to ChatML Dataset Pipeline

This project converts PDF books into ChatML training data using a two-stage workflow:

1. **Stage 1 (pagewise):** Generate grounded QnA from single pages.
2. **Stage 2 (cross-page):** Build evidence packs and synthesize multi-page QnA.

The design focuses on reliability: strict JSON outputs, quality gates, detailed logs, and checkpoint/resume support.

## What This Produces

- `output/chatml_dataset.jsonl`: main dataset output
- `output/cross_page_chatml_dataset.jsonl`: cross-page records only
- `output/page_artifacts.jsonl`: Stage 1 accepted page artifacts
- `output/skipped_pages.jsonl`: unusable/blank page log
- `output/quality_log.jsonl`: Stage 1 quality decisions
- `output/cross_page_quality_log.jsonl`: Stage 2 quality decisions
- `output/prompt_log.jsonl`: prompt trace (optional)
- `output/api_metrics.jsonl`: API latency/token/cost telemetry
- `output/process_log.jsonl`: process lifecycle events
- `output/analytics_report.json`: run analytics
- `output/token_stats.json`: token summary by call type
- `output/run_checkpoint.json`: checkpoint state for resume
- `output/failed_writes.jsonl`: fallback write-failure log

## How It Works

### Stage 1: Local Grounding

- Render each PDF page to an image data URL.
- Prompt the model to generate QnA from that single page.
- Retry blank/unreadable pages at higher DPI when configured.
- Run quality filters:
  - duplicate and near-duplicate rejection
  - model critique (optional)
  - citation checks against extracted page text
- Write accepted local records to ChatML.
- Save per-page artifacts for Stage 2.

### Stage 2: Cross-Page Synthesis

- Build deterministic evidence packs from Stage 1 artifacts using:
  - adjacent page windowing
  - lexical overlap
  - heading continuity
- Synthesize cross-page QnA from evidence packs.
- Require evidence-based multi-page reasoning in output schema.
- Run cross-page quality checks for:
  - grounding
  - usefulness
  - true multi-page dependency
  - quote/page consistency
  - duplicate similarity
- Write cross-page results to a separate output file, and optionally merge into main output.

## Quick Start

1) Create and activate a virtual environment.

2) Install dependencies:

```bash
pip install -r requirements.txt
```

3) Create env file:

```bash
copy .env.example .env
```

Then set your key in `.env`:

```env
OPENAI_API_KEY=...
```

4) Create config:

```bash
copy config.example.yaml config.yaml
```

5) Put PDF files in `books/`.

6) Run a cheap smoke test:

```bash
python run.py --config config.yaml --dry-run
```

7) Optional prompt preview:

```bash
python show_prompts.py --config config.yaml --page 1
```

8) Run full pipeline:

```bash
python run.py --config config.yaml
```

9) Resume from checkpoint:

```bash
python run.py --config config.yaml --resume
```

10) Optional mode flags:

```bash
python run.py --config config.yaml --skip-cross-page
python run.py --config config.yaml --cross-page-only --resume
```

## Fast Validation Checklist

After a test run, verify:

- `output/chatml_dataset.jsonl` exists
- `output/analytics_report.json` exists
- `output/process_log.jsonl` contains `run_started` and `run_completed`
- if cross-page is enabled, `output/page_artifacts.jsonl` exists

## Configuration Areas

### Core Runtime

- `runtime.model`
- `runtime.max_pages_per_book`
- `runtime.dpi`
- `runtime.retry_dpi`
- `runtime.max_unusable_retries`
- `runtime.request_timeout_seconds`
- `runtime.parallel_critique_workers`
- `runtime.parallel_future_timeout_seconds`

### Stage 1 Dataset

- `dataset.output_path`
- `dataset.qas_per_page`
- `dataset.user_profile`
- `dataset.variety`
- `dataset.citation`

### Stage 1 Quality

- `quality.enabled`
- `quality.use_model_self_critique`
- `quality.critique_model`
- `quality.min_grounding_score`
- `quality.min_usefulness_score`
- `quality.duplicate_similarity_threshold`
- `quality.require_citation_match_if_text_available`

### Stage 2 Cross-Page

- `cross_page.enabled`
- `cross_page.min_pages_per_pack`
- `cross_page.max_pages_per_pack`
- `cross_page.pack_overlap_window`
- `cross_page.max_cross_page_qas_per_pack`
- `cross_page.max_evidence_quotes_per_item`
- `cross_page.use_local_qas_as_hints`
- `cross_page.synthesis_model`
- `cross_page.synthesis_temperature`
- `cross_page.output_path`
- `cross_page.artifact_path`
- `cross_page.quality_log_path`
- `cross_page.require_quote_match_if_text_available`
- `cross_page.min_cross_page_grounding_score`
- `cross_page.min_cross_page_usefulness_score`
- `cross_page.min_multi_page_score`
- `cross_page.merge_into_final_dataset`

## Output Interpretation

- `local_candidates`: raw Stage 1 generated items before final filtering
- `local_accepted`: Stage 1 records that passed quality gate
- `cross_page_candidates`: raw Stage 2 generated items before final filtering
- `cross_page_accepted`: Stage 2 records that passed cross-page quality gate
- final dataset count is usually lower than candidate counts due to filtering

## Project Structure

- `run.py`: CLI entrypoint
- `show_prompts.py`: prompt preview utility
- `multimodal_dataset/pipeline.py`: main orchestration
- `multimodal_dataset/config.py`: typed config loader
- `multimodal_dataset/openai_client.py`: generation API wrapper
- `multimodal_dataset/quality.py`: quality logic + critique wrappers
- `multimodal_dataset/pdf_pages.py`: rendering and text extraction
- `multimodal_dataset/chatml.py`: record formatting and JSONL append
- `multimodal_dataset/analytics.py`: analytics writer
- `multimodal_dataset/page_artifacts.py`: Stage 1 page artifact builder
- `multimodal_dataset/evidence_packs.py`: deterministic evidence pack builder
- `multimodal_dataset/synthesis.py`: Stage 2 synthesis wrapper/schema

## Notes

- Use `--dry-run` before longer runs.
- Keep `.env` private and never commit API keys.
- PDF files in `books/` are ignored by git.
- Cost metrics are estimates based on configured token prices.
- `run_id` is attached to accepted records for traceability.
- Checkpoint/resume is designed for long-running jobs.
# Multimodal PDF -> ChatML Dataset Pipeline

This project validates a simple idea:

- Put PDF books in a folder.
- Provide your OpenAI key in `.env`.
- Tune a YAML recipe for QnA variety and style.
- Run a pipeline that uses a multimodal model on page images.
- Get `ChatML` JSONL records for instruction fine-tuning.

## What you get

- `output/chatml_dataset.jsonl`: final ChatML-style training records
- `output/skipped_pages.jsonl`: skipped blank/unreadable page audit log
- `output/prompt_log.jsonl`: exact request prompts sent to the model (optional)
- `output/api_metrics.jsonl`: per-call latency, token usage, and estimated cost
- `output/quality_log.jsonl`: per-item accept/reject decisions from quality gate
- `output/analytics_report.json`: run-level dataset analytics
- `output/token_stats.json`: token totals split by generation vs judge calls
- `output/run_checkpoint.json`: crash-safe checkpoint for resume
- `output/failed_writes.jsonl`: fallback log when a file write fails
- `output/process_log.jsonl`: end-to-end process events and per-call success logs
- `output/page_artifacts.jsonl`: Stage-1 accepted page artifacts for synthesis
- `output/cross_page_chatml_dataset.jsonl`: Stage-2 cross-page ChatML records
- `output/cross_page_quality_log.jsonl`: Stage-2 quality decisions

## Why this avoids OCR pain

The pipeline renders each PDF page as an image and sends it directly to a multimodal model. This bypasses explicit OCR scripts and keeps extraction logic simple.

## Architecture (current state)

### High-level flow

1) **Config + runtime bootstrap**
- Loads `config.yaml` into typed config objects.
- Loads API key from `.env`.
- Resolves matching PDFs from `input.books_dir` + `input.glob`.
- Initializes `run_id`, checkpoint state, and append/truncate behavior.

2) **Page ingestion**
- Converts each PDF page to an image data URL (`dpi`-controlled).
- Runs generation prompt on each page.
- If page is `blank`/`unreadable`, retries once at `retry_dpi`.

3) **Generation stage**
- Calls multimodal model with strict JSON schema output.
- Collects per-call metrics: latency, input/output tokens, cost estimate.
- Emits process events and API metrics logs.

4) **Quality gate stage**
- Filters exact/near-duplicates before critique.
- Runs judge-model critique (optionally in parallel workers).
- Applies acceptance rules:
  - grounding threshold
  - usefulness threshold
  - citation-to-page-text match (when text is extractable)
- Accepts/rejects each candidate with full audit logs.

5) **Persistence + recovery**
- Writes accepted records to ChatML JSONL.
- Updates checkpoint after each processed page.
- Uses append-mode and failed-write fallback logging for resilience.

6) **Run finalization**
- Writes analytics and token summary reports.
- Marks checkpoint status as completed.
- Emits final summary to console + process log.

### Module responsibilities

- `run.py`
  - Thin CLI entrypoint for the pipeline.
- `show_prompts.py`
  - Prompt preview helper from config without running generation.
- `multimodal_dataset/config.py`
  - Typed config schema + YAML loader.
- `multimodal_dataset/pdf_pages.py`
  - PDF page rendering, text extraction, and page-count utilities.
- `multimodal_dataset/openai_client.py`
  - Generation API call wrapper + usage/cost metric extraction.
- `multimodal_dataset/quality.py`
  - Normalization/similarity, citation checks, usefulness heuristic, judge call.
- `multimodal_dataset/chatml.py`
  - ChatML record formatting + durable JSONL append (`fsync`).
- `multimodal_dataset/analytics.py`
  - Writes run-level analytics report JSON.
- `multimodal_dataset/pipeline.py`
  - End-to-end orchestration, logging, quality filtering, checkpointing, resume.

### Logging and observability model

- **Prompt trace**: `output/prompt_log.jsonl`
- **API telemetry**: `output/api_metrics.jsonl`
- **Quality decisions**: `output/quality_log.jsonl`
- **Process lifecycle events**: `output/process_log.jsonl`
- **Skipped pages**: `output/skipped_pages.jsonl`
- **Fallback write failures**: `output/failed_writes.jsonl`
- **Run analytics**: `output/analytics_report.json`
- **Token breakdown**: `output/token_stats.json`
- **Recovery state**: `output/run_checkpoint.json`

### Parallelism and safety

- Parallelism is applied to **judge/critique calls per page** via worker pool.
- Each critique task uses its own client instance (avoids shared-client contention).
- Futures have a page-level timeout; timed-out tasks are cancelled and rejected.
- Checkpoint writes are atomic (`.tmp` + replace).
- JSONL writes are append + flush + fsync for better crash resilience.

### Stage 2: hierarchical cross-page synthesis

- Stage 1 local generation and quality gate remain unchanged.
- Accepted local page outputs are materialized as `page_artifacts.jsonl`.
- Deterministic evidence packs are built from related pages (adjacency + overlap + heading continuity).
- Stage 2 synthesis generates only evidence-grounded multi-page items.
- A dedicated cross-page quality gate validates grounding, usefulness, and multi-page dependency.
- Cross-page outputs are written separately and can also be merged into the final dataset.

## Quick start

1) Create and activate virtual environment.

2) Install dependencies:

```bash
pip install -r requirements.txt
```

3) Set env file:

```bash
copy .env.example .env
```

Then edit `.env` and add:

```env
OPENAI_API_KEY=...
```

4) Copy config template:

```bash
copy config.example.yaml config.yaml
```

5) Add PDFs into `books/`.

6) Run cheap validation first:

```bash
python run.py --config config.yaml --dry-run
```

7) Preview prompts before spending tokens (recommended):

```bash
python show_prompts.py --config config.yaml --page 1
```

8) Run full pipeline:

```bash
python run.py --config config.yaml
```

9) Resume a stopped/crashed run from checkpoint:

```bash
python run.py --config config.yaml --resume
```

10) Optional mode flags:

```bash
python run.py --config config.yaml --skip-cross-page
python run.py --config config.yaml --cross-page-only --resume
```

Output defaults to `output/chatml_dataset.jsonl`.
Skipped or unusable pages are logged to `output/skipped_pages.jsonl`.
Prompt requests are logged to `output/prompt_log.jsonl` when `runtime.log_prompts` is true.
API telemetry is logged to `output/api_metrics.jsonl` when `runtime.log_api_metrics` is true.
Quality decisions are logged to `output/quality_log.jsonl` when `quality.enabled` is true.
Analytics are written to `output/analytics_report.json` after each run.
Token statistics are written to `output/token_stats.json` after each run.
Checkpoint state is updated at `output/run_checkpoint.json` after each page.
The pipeline logs a pre-run sample estimate at startup (`run_started` event).
Parallel critique futures have a safety timeout; stalled tasks are cancelled and marked failed.

## Typical workflow for scanned books

1) Start with `--dry-run` to process only 1 page/book.
2) If output quality is weak, increase `runtime.dpi` and `runtime.retry_dpi`.
3) Keep `dataset.qas_per_page` low (for example 3-5) during iteration.
4) Inspect `prompt_log.jsonl` and refine prompt/config before full runs.
5) Scale `runtime.max_pages_per_book` after quality is stable.
6) Inspect `analytics_report.json` for latency/cost trends before scaling.
7) Inspect `process_log.jsonl` for per-call success/failure traces.

## Interpreting output counts

- `total_candidate_qas` in `analytics_report.json` is raw model output before filtering.
- Final dataset count is lower because quality gate + duplicate filtering remove weak/redundant items.
- With `runtime.append_mode: true`, output files accumulate across runs.
- For clean per-run outputs, set `runtime.append_mode: false` or write to a new output folder.

## Config knobs you can tune

- `dataset.qas_per_page`: how many QnA pairs per page
- `dataset.user_profile`: style targeting for answers
- `dataset.variety`: question types, difficulty mix, answer styles
- `dataset.citation`: citation quote requirements
- `runtime.max_pages_per_book`: cost-control lever
- `runtime.retry_dpi`: higher DPI used for unreadable/blank retry
- `runtime.max_unusable_retries`: retry count for unusable pages
- `runtime.model`: swap in stronger/cheaper multimodal model
- `runtime.log_prompts`: enable/disable prompt request logging
- `runtime.prompt_log_path`: where prompt request logs are written
- `runtime.log_api_metrics`: enable/disable API telemetry logging
- `runtime.api_metrics_log_path`: where API telemetry logs are written
- `runtime.generation_input_cost_per_1m_tokens_usd`: generation input price for cost estimate
- `runtime.generation_output_cost_per_1m_tokens_usd`: generation output price for cost estimate
- `runtime.judge_input_cost_per_1m_tokens_usd`: critique input price for cost estimate
- `runtime.judge_output_cost_per_1m_tokens_usd`: critique output price for cost estimate
- `runtime.checkpoint_enabled`: enable page-level checkpoint persistence
- `runtime.checkpoint_path`: checkpoint file path for resume support
- `runtime.append_mode`: append logs instead of truncating output files
- `runtime.failed_writes_log_path`: fallback path for failed write recovery
- `runtime.parallel_critique_workers`: parallel workers for judge/critique calls
- `runtime.parallel_future_timeout_seconds`: max wait for a page's parallel critique futures
- `runtime.process_log_path`: process-level event log path
- `runtime.verbose_success_logs`: print per-event success logs to console
- `dataset.skipped_pages_output_path`: audit log for skipped pages
- `quality.enabled`: enable/disable post-generation quality gate
- `quality.use_model_self_critique`: model-based QA critique pass
- `quality.critique_model`: judge model (can be stronger than `runtime.model`)
- `quality.min_grounding_score`: grounding acceptance threshold
- `quality.min_usefulness_score`: usefulness acceptance threshold
- `quality.duplicate_similarity_threshold`: near-duplicate rejection threshold
- `quality.quality_log_path`: per-item quality decision log
- `analytics.report_path`: output location for analytics summary JSON
- `analytics.token_stats_path`: output location for token summary JSON
- `cross_page.enabled`: enable/disable Stage-2 cross-page synthesis
- `cross_page.min_pages_per_pack` / `max_pages_per_pack`: evidence pack sizing
- `cross_page.max_cross_page_qas_per_pack`: synthesis item cap per pack
- `cross_page.use_local_qas_as_hints`: use local QAs only as optional hints
- `cross_page.synthesis_model`: model used for Stage-2 synthesis
- `cross_page.output_path`: cross-page dataset output path
- `cross_page.artifact_path`: page artifact output path
- `cross_page.quality_log_path`: cross-page quality decisions output path
- `cross_page.min_cross_page_grounding_score`: cross-page grounding threshold
- `cross_page.min_cross_page_usefulness_score`: cross-page usefulness threshold
- `cross_page.min_multi_page_score`: threshold for true multi-page reasoning
- `cross_page.merge_into_final_dataset`: also append cross-page records into main dataset

## Project structure

- `run.py`: main pipeline entrypoint
- `show_prompts.py`: prompt preview utility
- `multimodal_dataset/pipeline.py`: orchestration and retry/skip logic
- `multimodal_dataset/openai_client.py`: multimodal API request + JSON schema output
- `multimodal_dataset/pdf_pages.py`: PDF page rendering to image data URLs
- `multimodal_dataset/quality.py`: quality scoring and duplicate utilities
- `multimodal_dataset/analytics.py`: analytics report writer
- `multimodal_dataset/page_artifacts.py`: stage-1 page artifact construction
- `multimodal_dataset/evidence_packs.py`: deterministic evidence pack builder
- `multimodal_dataset/synthesis.py`: cross-page synthesis API wrapper/schema
- `config.yaml`: your local runnable config
- `config.example.yaml`: template config for sharing

## Notes

- Use `--dry-run` to test quality and estimate spend.
- Keep your real `.env` private (`.gitignore` already excludes it).
- Keep private/copyrighted PDFs out of git (`books/*.pdf` is ignored).
- For production quality, add rejection/critique passes and dedup filters.
- This pipeline includes a quality gate (self-critique, citation grounding check, dedup).
- Cost estimates are approximate and depend on your configured token prices.
- Each accepted ChatML record includes `run_id` metadata for run-level traceability.
- Use `--resume` with `checkpoint_enabled: true` for long full-book runs.
- Final output count is typically lower than raw candidate count because quality gate and dedup filters reject weak items.
- The model now classifies each page (`usable`, `blank`, `unreadable`, `index_only`, `image_only`) and returns `items: []` for blank/unreadable pages.
