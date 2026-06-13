# Evaluation pipeline — operator guide

This is the *runbook* for the dataset prep and eval workflow described in
[`../eval_plan.md`](../eval_plan.md). Treat that file as the strategy and
this file as the recipe.

All scripts live under `../scripts/`. Run them from the project root.

---

## Folder layout produced by this pipeline

```
corpus/
  raw/                     PDFs as downloaded from arXiv (+ manifest.jsonl)
  train/                   training split (hardlinked from raw)
  holdout/                 held-out split (hardlinked from raw, never trained on)
  manifest.jsonl           combined manifest with `split` column
eval/
  custom/
    candidates.csv         human-review surface (you fill q/a/status here)
    page_texts.jsonl       cached page text for the eval runner
    test.jsonl             FINAL eval set after finalize step
    results/<tag>/         per-model predictions and scores
  baselines/<tag>/         lm-evaluation-harness output
```

---

## Step 0 — install dependencies

The existing `requirements.txt` covers doc2instruct's pipeline. The eval
scaffolding needs a few extras. Install once:

```bash
pip install arxiv tqdm
# only needed when you actually run the harness on a GPU host:
# pip install "lm-eval[hf]"
# pip install transformers accelerate bitsandbytes peft
```

`PyMuPDF` (`fitz`) and `openai` are already in `requirements.txt`.

---

## Step 1 — download the arXiv corpus

```bash
python scripts/download_arxiv.py --limit 150
```

What it does:
- Pulls 150 most-recent papers from `cs.CL`, `cs.LG`, `cs.AI` since 2023-01-01.
- Writes `corpus/raw/<arxiv_id>.pdf` and appends to `corpus/raw/manifest.jsonl`.
- Idempotent — re-running with the same `--limit` resumes from where it left off.

Smoke first (5 papers, ~30 seconds):

```bash
python scripts/download_arxiv.py --limit 5 --output-dir corpus/raw
```

Useful flags:
- `--categories cs.CL cs.LG`   narrower corpus
- `--start-date 2024-06-01`     more recent papers only
- `--verbose`                    see arXiv API queries

---

## Step 2 — split into train / holdout

```bash
python scripts/split_corpus.py
```

What it does:
- Reads `corpus/raw/manifest.jsonl`.
- Deterministically (seed=42) sets aside 15 papers as the held-out set.
- Hardlinks PDFs into `corpus/train/` and `corpus/holdout/`.
- Writes `corpus/manifest.jsonl` with a `split` column.

The script refuses to clobber existing splits unless you pass `--force` —
that's intentional, because re-splitting after you've already trained
silently contaminates your held-out set.

---

## Step 3 — bootstrap the custom eval candidates

```bash
# blank candidates (no API cost) — recommended for full control:
python scripts/bootstrap_eval_set.py

# or, draft candidates with an LLM (costs $; review carefully):
python scripts/bootstrap_eval_set.py --use-openai --openai-model gpt-4o-mini
```

Output:
- `eval/custom/candidates.csv` — open in Excel / LibreOffice / VS Code CSV editor.
- `eval/custom/page_texts.jsonl` — full page text, used later by the eval runner.

Each row in the CSV has:
- `paper_id`, `arxiv_id`, `question_type` (single_page or cross_page), `pages`
- `page_text_excerpt` — first 1500 chars of the relevant page(s) for inline reference
- `candidate_question`, `candidate_answer`, `evidence_quote` — fill these in
- `status` — set to `accept` (keep), `edit` (kept after you edited), or `reject`
- `notes` — anything you want to remember

Cross-page volume: the bootstrapper now generates at least `--cross-ratio`
(default 0.5) cross-page candidates per paper relative to single-page ones,
drawing adjacent page pairs first then widening the gap up to `--cross-max-gap`.
So the review pool supports a genuinely multi-page eval, not a single-page one.

Target: ~150 accepted rows total, with **cross-page at least ~50% of single-page**.
Quality > quantity. 100 well-written items beat 300 sloppy ones.

### Recommended: review in the notebook

Instead of editing the CSV by hand, open the interactive reviewer:

```bash
pip install ipywidgets pandas   # one-time (jupyter too if you don't have it)
jupyter lab eval/review_candidates.ipynb   # or: jupyter notebook
```

It shows the full source page text beside each candidate, lets you edit the
question/answer/evidence and set `status`, autosaves to `candidates.csv`, tracks
the cross/single ratio live, and has a Finalize cell that writes `test.jsonl`.

---

## Step 4 — finalize the test set

After you've reviewed and saved the CSV:

```bash
python scripts/finalize_eval_set.py
```

What it does:
- Reads `eval/custom/candidates.csv`.
- Keeps rows where `status` is `accept` or `edit`.
- Validates each row (non-empty question/answer/evidence, valid pages).
- Writes `eval/custom/test.jsonl`.

If validation drops rows, the script logs why. Fix in the CSV and re-run.

---

## Step 5 — record baseline numbers (BEFORE any fine-tuning)

This step needs a GPU. Run on Kaggle (T4×2) or RunPod (RTX 4090).

```bash
# Tier 1 + Tier 2 standard benchmarks
python scripts/run_baselines.py \
    --model-path Qwen/Qwen2.5-7B-Instruct \
    --tag base_qwen25_7b

# Tier 3 custom held-out eval
python scripts/run_custom_eval.py \
    --backend hf \
    --model-path Qwen/Qwen2.5-7B-Instruct \
    --tag base_qwen25_7b
```

Both write to `eval/<...>/<tag>/`. **You need these baseline numbers to claim
any improvement later.** Don't fine-tune before you have them.

For a fast sanity smoke (5–10 minutes):
```bash
python scripts/run_baselines.py --model-path Qwen/Qwen2.5-7B-Instruct \
    --tag smoke_base --tasks gsm8k_cot --limit 50
```

---

## Step 6 — generate training data with doc2instruct

This uses your existing pipeline:

```bash
# point doc2instruct at the train split
python run.py --config config.yaml
```

Make sure `config.yaml`'s `input.books_dir` is set to `corpus/train/` (NOT
`corpus/raw/`) and `input.glob` is `*.pdf`. Otherwise you contaminate the
holdout. A ready-made `config.train.yaml` is provided for exactly this — run:

```bash
python run.py --config config.train.yaml
```

---

## Step 7 — fine-tune (Kaggle / RunPod)

Out of scope for this scaffolding — handled in the Unsloth notebook step
described in `eval_plan.md`. The output of fine-tuning is a model directory
or HF repo id you'll feed into Step 8.

---

## Step 8 — re-run baselines and custom eval on the fine-tune

```bash
python scripts/run_baselines.py \
    --model-path /path/to/finetuned_model \
    --tag stage1_qwen25_7b

python scripts/run_custom_eval.py \
    --backend hf \
    --model-path /path/to/finetuned_model \
    --tag stage1_qwen25_7b
```

Compare `eval/baselines/base_qwen25_7b/summary.json` against
`eval/baselines/stage1_qwen25_7b/summary.json` — that's your headline result.

For the Stage 1 vs Stage 1 + Stage 2 ablation, repeat with a model trained
on the cross-page-augmented dataset and tag it `stage12_qwen25_7b`.

---

## Sanity checks worth doing

- After Step 2: `wc -l corpus/manifest.jsonl` should equal the total paper count.
- After Step 4: `wc -l eval/custom/test.jsonl` should be your accepted-row count.
- Before Step 7: run the contamination gate (implements `eval_plan.md` §4.3):

```bash
python scripts/check_contamination.py
```

  It checks two things using `corpus/manifest.jsonl` as the authority on splits:
  (1) no held-out `arxiv_id` appears in the generated dataset metadata, and
  (2) no held-out paper's text overlaps the dataset via shingled hashing.
  A non-zero exit means STOP — do not fine-tune until it is clean.

---

## Troubleshooting

**arXiv downloads stall or fail intermittently.**
The arXiv API is rate-limited and occasionally flaky. The downloader retries 3×
and skips persistent failures. Re-run the same command — it resumes.

**`lm_eval` task name not found.**
Task IDs evolve. `run_baselines.py` pre-flights `lm_eval --tasks list` and skips
any task it can't find (logging a warning) instead of aborting the whole run, so
one bad name no longer sinks the suite. To find a current name manually, run
`lm_eval --tasks list | grep -i squad` and update `DEFAULT_TASKS`.

**HotpotQA (the multi-hop / Stage-2 benchmark).**
HotpotQA is not native to lm-eval, so we ship a real-dataset task config at
`eval/lm_eval_tasks/hotpotqa.yaml` (distractor setting, answer EM/F1).
`run_baselines.py` registers it automatically via `--include-path eval/lm_eval_tasks`.
It needs network access to pull the `hotpot_qa` dataset on first run and may
require `trust_remote_code` (already set in the YAML). If it can't load, the
pre-flight drops it and the other tasks still run.

**OOM during `run_custom_eval.py` HF backend.**
Add `--model-args load_in_4bit=true` (you'll need bitsandbytes installed),
or switch to the `openai` backend pointed at a vllm server.

**Custom eval scores are zero.**
Check `eval/custom/results/<tag>/predictions.jsonl` — usually the model is
producing verbose chain-of-thought that wraps the actual answer. Either tighten
the system prompt in `run_custom_eval.py` or rely on the `--judge` flag
for soft scoring.
