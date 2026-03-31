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

## Why this avoids OCR pain

The pipeline renders each PDF page as an image and sends it directly to a multimodal model. This bypasses explicit OCR scripts and keeps extraction logic simple.

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

## Project structure

- `run.py`: main pipeline entrypoint
- `show_prompts.py`: prompt preview utility
- `multimodal_dataset/pipeline.py`: orchestration and retry/skip logic
- `multimodal_dataset/openai_client.py`: multimodal API request + JSON schema output
- `multimodal_dataset/pdf_pages.py`: PDF page rendering to image data URLs
- `multimodal_dataset/quality.py`: quality scoring and duplicate utilities
- `multimodal_dataset/analytics.py`: analytics report writer
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
