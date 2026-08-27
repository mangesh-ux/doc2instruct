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
    candidates.csv              first-pass drafted candidates
    candidates_v3.csv           rewritten under the strict rubric (step 3a)
    cross_page_candidates.csv   multi-hop drafts, one span per page (step 3b)
    cross_page_ablated.csv      + single-page ablation verdicts (step 3c)
    candidates_verified.csv     after the mechanical gate + selection (step 3d)
    page_texts.jsonl            cached page text for the eval runner
    test.jsonl                  FINAL eval set
    test_provenance.json        how the set was built, per-item review status
    verification_report.json    every check and every rejection reason
    ablation_report.json         multi-hop shortcut rate
    DATASET_CARD.md             composition, method, limitations — read this
    results/<tag>/              per-model predictions and scores
  baselines/<tag>/              lm-evaluation-harness output
  finetune_qlora.ipynb          QLoRA fine-tuning (GPU)
  review_candidates.ipynb       interactive human review
  RESULTS.md                    results table template
output/
  smoke_run/                    2-paper validation run (never publish)
  train_run/                    the real generation run
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

### Step 3a — upgrade the candidates (strict rubric)

The first-pass candidates are weak: generic "main contribution" questions, and
evidence quotes that are paraphrases rather than copies. Rewrite them:

```bash
python scripts/improve_candidates.py --workers 8 --out-csv eval/custom/candidates_v3.csv
```

This rewrites every `pending` row with `gpt-4.1` under a rubric that demands a
specific verifiable answer, a short scoreable answer (never yes/no), and
evidence copied character-for-character. Rows on unusable pages (references,
figure-only) come back `status=reject` so you never look at them.

> If you change the rubric, re-check the *verbatim* rule. Telling the model the
> quote must be "richer than the answer and not a copy of it" makes it
> paraphrase — that alone dropped the usable pool from 269 to 123 items.

### Step 3b — build genuinely multi-hop cross-page items

A `cross_page` row with one evidence quote is not a multi-hop item: the quote
sits on one page, so the question is single-page answerable. 57 of 58 items in
the first build had this problem. Rebuild with one span *per page*:

```bash
python scripts/build_cross_page_eval.py --enumerate-pairs --workers 8
```

Draft several times your target — the ablation below rejects most of them.

### Step 3c — prove they are multi-hop (the ablation)

```bash
python scripts/ablate_cross_page.py --workers 8
```

Answers each question given only page A, then only page B, and keeps the item
only if neither single page reaches token F1 ≥ 0.6 against the gold answer.
**Expect to lose ~60%** — that is the measured shortcut rate among items the
generator itself certified as multi-hop. Results in `ablation_report.json`.

### Step 3d — mechanical verification gate

```bash
python scripts/verify_candidates.py \
    --in-csv eval/custom/cross_page_ablated.csv \
             eval/custom/cross_page_ablated_wide.csv \
             eval/custom/candidates_v3.csv \
    --target-single 90 --target-cross 60 --require-ablation-for-cross
```

Checks evidence is verbatim in the source page (unicode/hyphenation tolerant,
coverage ≥ 0.85), that each cross-page span is verbatim on *its own* page,
rejects generic and yes/no items, dedupes, then fills the type quotas
round-robin across papers so no paper dominates. Reasons for every rejection
land in `verification_report.json`.

List your most trusted CSV first — earlier files win ties.

### Step 3e — human spot-check in the notebook

The gate is mechanical; it cannot tell you a question is *interesting*.

```bash
pip install ipywidgets pandas   # one-time (jupyter too if you don't have it)
jupyter lab eval/review_candidates.ipynb   # or: jupyter notebook
```

It shows the full source page text beside each candidate, lets you edit the
question/answer/evidence and set `status`, autosaves, and tracks the
cross/single ratio live.

**This step is what lets you call the set hand-verified.** Until an item is
touched here its provenance says `machine_gate`, and `finalize_eval_set.py`
warns. Spot-check ~30 spread across papers and both question types.

---

## Step 4 — finalize the test set

```bash
python scripts/finalize_eval_set.py
```

What it does:
- Reads `eval/custom/candidates_verified.csv`.
- Keeps rows where `status` is `accept` or `edit`.
- Validates each row (non-empty question/answer/evidence, valid pages).
- Writes `eval/custom/test.jsonl` plus `test_provenance.json`, and stamps every
  item with its evidence coverage, which checks it passed, and whether a human
  reviewed it.

If validation drops rows, the script logs why. Fix in the CSV and re-run.

Read `eval/custom/DATASET_CARD.md` before quoting any number from this set — its
limitations section is not boilerplate.

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

**Smoke-test first. Always.** The full run is ~11.6 hours; a config or prompt
problem found at hour 10 costs a day. `config.smoke.yaml` runs 2 papers × 5
pages into `output/smoke_run/` in about 3 minutes:

```bash
python run.py --config config.smoke.yaml
```

Then check `output/smoke_run/analytics_report.json`:
- `local_accepted / total_candidate_qas` should be roughly 0.5–0.7. If it is
  near 0.2, the citation gate is rejecting good items — see `eval_plan.md` §8.1.
- `cross_page_accepted` must be **> 0**. Zero means Stage 2 is broken, not that
  your papers lack cross-page structure.
- `local_rejection_reasons` dominated by `citation_not_found_in_page_text` means
  a quote-matching problem, not a model-quality problem. The per-item
  `citation_coverage` in `quality_log.jsonl` tells you which.

Then the real run:

```bash
python run.py --config config.train.yaml     # add --resume after an interruption
```

`config.train.yaml` pins `input.books_dir` to `corpus/train` and namespaces
**all** outputs under `output/train_run/`. Both matter: pointing at
`corpus/raw/` contaminates the holdout, and writing to bare `output/` appends to
whatever an earlier run left there (`output/chatml_dataset.jsonl` still holds 53
records from an unrelated scanned book).

Measured expectations for 135 papers / 2,686 pages: ~9,100 records, ~11.6 h,
~$90 at gpt-4.1 list price (~$199 at the config's conservative cost constants).
Pages are generated serially — `parallel_critique_workers` only parallelises the
judge — so time is the binding constraint.

### Then verify the split held

```bash
python scripts/check_contamination.py --dataset output/train_run/chatml_dataset.jsonl
```

Must exit 0 before you fine-tune. Exit 2 means held-out material leaked; stop
and regenerate.

---

## Step 7 — fine-tune (Kaggle / RunPod)

Use `eval/finetune_qlora.ipynb` — QLoRA on Qwen 2.5 7B via Unsloth, needs a
~16 GB GPU. Run it twice, changing only `RUN_NAME`:

- `RUN_NAME = "stage1"` — trains on `chatml_dataset.jsonl`
- `RUN_NAME = "stage1_stage2"` — adds `cross_page_chatml_dataset.jsonl`

The base model needs no run; it is evaluated untuned. The notebook fingerprints
its input data (SHA256 + record counts), re-checks contamination at the point of
use, audits sequence lengths for silent truncation, and writes
`run_manifest.json` next to the adapter so any result traces back to the exact
data and hyperparameters that produced it.

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
