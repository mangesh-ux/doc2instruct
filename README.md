# Multimodal PDF -> ChatML Dataset Pipeline

This project validates a simple idea:

- Put PDF books in a folder.
- Provide your OpenAI key in `.env`.
- Tune a YAML recipe for QnA variety and style.
- Run a pipeline that uses a multimodal model on page images.
- Get `ChatML` JSONL records for instruction fine-tuning.

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

7) Run full pipeline:

```bash
python run.py --config config.yaml
```

Optional: preview prompts before spending tokens:

```bash
python show_prompts.py --config config.yaml --page 1
```

Output defaults to `output/chatml_dataset.jsonl`.
Skipped or unusable pages are logged to `output/skipped_pages.jsonl`.
Prompt requests are logged to `output/prompt_log.jsonl` when `runtime.log_prompts` is true.

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
- `dataset.skipped_pages_output_path`: audit log for skipped pages

## Notes

- Use `--dry-run` to test quality and estimate spend.
- Keep your real `.env` private (`.gitignore` already excludes it).
- For production quality, add rejection/critique passes and dedup filters.
- The model now classifies each page (`usable`, `blank`, `unreadable`, `index_only`, `image_only`) and returns `items: []` for blank/unreadable pages.
