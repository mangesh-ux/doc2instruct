# doc2instruct — My Understanding (for review)

This is my working understanding of the project after reading the full pipeline,
the eval scripts, configs, and planning docs. It is split into two halves:

1. **The data-synthesis pipeline** (what already exists and works).
2. **The evaluation plan** (what we're about to build/run, plus gaps I noticed).

At the end I list **open questions and risks** I want to confirm with you before
we start changing things. Please correct anything I got wrong.

---

## Part 1 — The synthesis pipeline (doc2instruct)

### 1.1 One-line summary
Convert long-form PDFs into grounded instruction-tuning data (ChatML), using a
**hierarchical two-stage** design that prioritizes reliability and inspectability
over novelty.

- **Stage 1 — local / single-page:** render each PDF page to an image, ask a
  multimodal model for grounded QnA, then run a quality gate.
- **Stage 2 — cross-page synthesis:** assemble deterministic "evidence packs"
  from related pages and synthesize multi-page QnA that a single page can't
  produce, then run a separate cross-page quality gate.

The whole thing is built for long, resumable, auditable runs.

### 1.2 End-to-end flow (as implemented in `multimodal_dataset/pipeline.py`)

1. **Bootstrap:** load `.env` (needs `OPENAI_API_KEY`), load `config.yaml` into
   typed dataclasses (`config.py`), resolve target PDFs via `input.books_dir` +
   `input.glob`, build/restore a `run_id`, optionally load a checkpoint.
2. **Stage 1 per page** (`_generate_with_retry`):
   - Render page → PNG data URL (`pdf_pages.py`, `dpi` controlled).
   - Call multimodal model with a **strict JSON schema** (`openai_client.py`),
     which forces a `page_status` (`usable | blank | unreadable | index_only |
     image_only`), a reason, and an `items[]` list of QA pairs.
   - If `page_status` is blank/unreadable, retry once at `retry_dpi`.
3. **Stage 1 quality gate** (`quality.py` + pipeline logic):
   - **Dedup first:** exact + near-duplicate (difflib ratio ≥
     `duplicate_similarity_threshold`) against `seen_pairs`.
   - **Model self-critique** (optional, parallelizable): a judge model returns
     `grounding_score`, `usefulness_score`, `grounded`, `useful`, `concerns`.
   - **Citation check:** if page text is extractable, the `citation_quote` must
     appear in the extracted text (`has_citation_match`).
   - **Thresholds:** `min_grounding_score`, `min_usefulness_score`.
   - Accepted items → ChatML record (`chatml.py`) appended to
     `output/chatml_dataset.jsonl` with `run_id` metadata.
4. **Stage 1 artifact:** for every processed page, write a durable
   `page_artifacts.jsonl` record (`page_artifacts.py`) holding page text,
   accepted QAs, a heading candidate, first non-empty lines, top keywords, and
   question-type distribution. This is the bridge to Stage 2.
5. **Stage 2 per book** (`evidence_packs.py` + `synthesis.py`):
   - Build packs from **usable pages that have accepted QAs**, using adjacent
     windowing (size `min/max_pages_per_pack`, `pack_overlap_window` overlap),
     plus lexical-overlap and heading-continuity signals.
   - Synthesize cross-page QnA with a strict schema: each item must include
     `source_pages` (≥2), `evidence_quotes` (page+quote), a `synthesis_type`
     enum, and `requires_multi_page_reasoning`. Page text is ground truth;
     local QAs are hints only.
   - **Cross-page quality gate** (`critique_cross_page_item`): grounding,
     usefulness, and a `multi_page_score` / `truly_multi_page` check, plus
     quote-consistency against pack text and dedup against both local and
     cross-page signatures.
   - Accepted → `output/cross_page_chatml_dataset.jsonl`, and optionally merged
     into the main dataset (`merge_into_final_dataset`).
6. **Finalize:** write `analytics_report.json`, `token_stats.json`, mark
   checkpoint `completed`, print a summary.

### 1.3 Reliability machinery (the part the project really cares about)
- **Checkpoint/resume:** atomic JSON writes; tracks processed pages, seen
  signatures, completed books, processed pack IDs, all counters and
  distributions. `--resume` continues; `--skip-cross-page` and
  `--cross-page-only` isolate stages.
- **Durable writes:** every append is flush + `fsync`; failed writes fall back
  to `failed_writes.jsonl`.
- **Telemetry:** prompt log, per-call API metrics (latency/tokens/cost),
  process lifecycle log, quality decision logs (local + cross-page),
  skipped-page log, analytics, token stats.
- **Parallelism:** only the Stage 1 critique calls are parallelized (per-page
  thread pool, per-task client, future timeout).

### 1.4 Key files
| File | Role |
|---|---|
| `run.py` | thin CLI → `pipeline.main` |
| `multimodal_dataset/pipeline.py` | orchestrator (the big one, ~1400 lines) |
| `multimodal_dataset/config.py` | typed config loader |
| `multimodal_dataset/openai_client.py` | Stage 1 generation + schema + cost |
| `multimodal_dataset/quality.py` | dedup/citation heuristics + both critique calls |
| `multimodal_dataset/pdf_pages.py` | render page → image, extract text |
| `multimodal_dataset/page_artifacts.py` | Stage 1 artifact + signal extraction |
| `multimodal_dataset/evidence_packs.py` | deterministic pack builder |
| `multimodal_dataset/synthesis.py` | Stage 2 synthesis call + schema |
| `multimodal_dataset/chatml.py` | ChatML formatting + durable append |
| `multimodal_dataset/analytics.py` | report writer |

---

## Part 2 — The evaluation plan

Strategy lives in `eval_plan.md` (locked v1, 2026-05-10); the operator runbook
is `eval/README.md`. The eval scripts are in `scripts/`.

### 2.1 The thesis being tested
Does instruction-tuning on doc2instruct data make a base model better at
**grounded document QA** — especially the **cross-page (Stage 2) reasoning** —
without wrecking general capability?

### 2.2 Corpus
- arXiv `cs.CL` / `cs.LG` / `cs.AI`, from 2023-01-01, ~150 papers.
- **15 papers held out** entirely from training, used to build the custom eval.
- Tracked per paper: `arxiv_id`, `sha256`, `split` in `corpus/manifest.jsonl`.

### 2.3 Three-tier evaluation
- **Tier 1 (sanity / no-regression):** MMLU subset, GSM8K — confirm fine-tuning
  didn't damage general capability (target: within ~1–2 pts of base).
- **Tier 2 (document QA):** SQuAD 2.0, Qasper (in-domain for arXiv), HotpotQA
  (the multi-hop test that directly probes the Stage 2 hypothesis).
- **Tier 3 (custom held-out):** ~150 hand-verified QA pairs (~60% single-page,
  ~40% cross-page) from the 15 held-out papers, in doc2instruct's own format.

### 2.4 Models under test
Qwen 2.5 7B base → +Stage 1 fine-tune → +Stage 1+Stage 2 fine-tune. The
Stage 1→Stage 1+2 delta on HotpotQA / cross-page custom set is the headline.

### 2.5 The eval scripts (what each does)
| Script | Purpose | Cost |
|---|---|---|
| `download_arxiv.py` | pull papers + manifest with SHA256; idempotent/resumable | free |
| `split_corpus.py` | deterministic (seed=42) train/holdout split; refuses to clobber w/o `--force` | free |
| `bootstrap_eval_set.py` | extract page text, emit `candidates.csv` (blank or LLM-drafted) + `page_texts.jsonl` | free unless `--use-openai` |
| `finalize_eval_set.py` | validate accepted CSV rows → `eval/custom/test.jsonl` | free |
| `run_baselines.py` | wraps `lm-eval` for Tier 1+2 → `summary.json` | GPU |
| `run_custom_eval.py` | HF or OpenAI backend on Tier 3; EM, F1, optional LLM-judge | GPU/API |

Canonical order: download → split → bootstrap → manual review → finalize →
baselines (base) → doc2instruct on train split → fine-tune (off-host) →
baselines + custom eval on fine-tune.

### 2.6 How custom eval actually scores (important nuance)
`run_custom_eval.py` builds context by **feeding the gold page text(s) back to
the model** and asking the question. So Tier 3 is **open-book reading
comprehension given the correct context** — it measures grounding/answer
quality, *not* retrieval. Metrics are SQuAD-style EM/F1 (strict on free-form
answers) plus an optional LLM-judge (default `gpt-4o-mini`), which is likely the
most meaningful signal for verbose model outputs.

---

## Part 3 — Observations, gaps, and risks (please confirm)

These are the things I'd want to resolve as we work through the eval. None are
blockers to understanding; several are blockers to a *clean* result.

1. **Config is not pointed at the training corpus.** `config.yaml` currently has
   `input.books_dir: "./books"` and `glob` pinned to a single scanned book
   (`"The Mysteries of Life and Death (scan).pdf"`). For the eval we must point
   it at `corpus/train/` with glob `*.pdf`. Also note `eval/README.md` Step 6
   says "`input.folder`" but the real key is `input.books_dir` — minor doc bug.

2. **No automated contamination check exists yet.** `eval_plan.md` §4.3 promises
   a shingled-hashing script that verifies no held-out text leaks into training
   data. Only a *manual* grep-the-ids sanity check is documented. If we want the
   anti-contamination claim to hold, we should build that script.

3. **Generation grounds on page *images*; eval grounds on extracted *text*.**
   doc2instruct sends page images to a multimodal model, but `bootstrap_eval_set`
   and `run_custom_eval` use `fitz` text extraction. For born-digital arXiv PDFs
   this is fine (text is clean), but it's an inconsistency worth being explicit
   about — and it means image rendering is arguably overkill (and pricey) for
   this corpus.

4. **Cost could be significant.** `config.yaml` uses `gpt-4.1` for generation,
   critique, *and* synthesis, with a **per-item** critique call. At
   `qas_per_page: 4` that's roughly 1 generation + up to 4 critiques per page,
   times every page of ~135 training papers, plus Stage 2. We should estimate
   spend (the pipeline's own cost telemetry can help) before a full run.

5. **lm-eval task names may not resolve as written.** `DEFAULT_TASKS` uses
   `squad2`, `hotpotqa`, `qasper`. Some of these aren't standard lm-eval task
   IDs (qasper in particular). The README already flags this; we may need to map
   to the correct current task names or provide custom task configs.

6. **EM/F1 on free-form answers will look harsh.** Generative models wrap answers
   in prose; normalized EM will be near-zero and F1 noisy. Expect the LLM-judge
   to carry the Tier 3 story. Worth deciding the judge model up front
   (`eval_plan.md` §7 leaves it open).

7. **"Adjacent" packs are adjacency *within usable pages*, not raw page numbers.**
   `evidence_packs.py` sorts usable-with-QA pages and windows over that list, so
   a pack can join pages that aren't physically adjacent if pages between them
   were unusable. The `pack_id` records the true page numbers, so it's traceable,
   but it's a subtlety for interpreting cross-page provenance.

8. **Open decisions before the GPU step** (from `eval_plan.md` §7): LoRA
   rank/alpha, epoch count, HotpotQA full vs multi-hop subset, and judge model.

---

## Part 4 — Patches applied this session

These are committed to code (not just notes):

1. **`run_custom_eval.py` HF backend bug (critical).** It returned the *entire
   prompt* as the prediction because it decoded the full sequence and tried to
   string-strip a prompt that still contained special tokens (which the decode
   had removed), so the strip never matched. Now it decodes **only the newly
   generated tokens** (slice after `prompt_len`). Without this, every local-GPU
   EM/F1 number would have been garbage.
2. **`run_custom_eval.py` judge efficiency.** The OpenAI judge client was being
   constructed once *per item*. Now built once and reused.
3. **`run_baselines.py` task names.** `squad2` → `squadv2` (verified correct id);
   removed `hotpotqa` from defaults because it is **not** a native lm-eval task
   (would have aborted the whole run). Added a pre-flight `filter_available_tasks`
   that queries `lm_eval --tasks list` and skips/warns on unknown tasks instead
   of crashing. `qasper` confirmed valid (group → `qasper_bool`+`qasper_freeform`).
4. **`eval/README.md`** `input.folder` → `input.books_dir` doc bug fixed; added
   pointers to `config.train.yaml` and the new contamination gate.
5. **New `scripts/check_contamination.py`** — implements the `eval_plan.md` §4
   anti-contamination protocol that previously existed only as a promise: an
   exact held-out-id check plus a shingled-hash text-overlap check, writing
   `eval/contamination_report.json` and returning a non-zero exit on leakage.
6. **New `config.train.yaml`** — a known-good config pinned to `corpus/train/`
   (glob `*.pdf`) so the held-out papers are never fed to generation. Your
   existing `config.yaml` is untouched.

## Part 5 — Ordered runbook to publishable artifacts

Local (no GPU) steps you can do now:

1. `pip install -r requirements-eval.txt`
2. `python scripts/download_arxiv.py --limit 5 --output-dir corpus/raw_smoke`
   (smoke), then `python scripts/download_arxiv.py --limit 150`.
3. `python scripts/split_corpus.py` → produces `corpus/manifest.jsonl` and the
   train/holdout dirs.
4. `python scripts/bootstrap_eval_set.py` (or `--use-openai` to pre-draft), then
   hand-review `eval/custom/candidates.csv`, then
   `python scripts/finalize_eval_set.py` → `eval/custom/test.jsonl`.
5. **Cost check before the big generation run:** dry-run doc2instruct on the
   train split: `python run.py --config config.train.yaml --dry-run` and read
   `output/analytics_report.json` for the per-page cost, then extrapolate.
6. Full generation: `python run.py --config config.train.yaml` (use `--resume`
   if interrupted).
7. **Gate before any GPU spend:** `python scripts/check_contamination.py` — must
   exit 0.

GPU / cloud steps (Kaggle / RunPod):

8. Baselines on base model: `run_baselines.py` + `run_custom_eval.py` with tag
   `base_qwen25_7b`. **Record these before fine-tuning.**
9. Fine-tune Qwen 2.5 7B (Stage 1 only, then Stage 1+2) via Unsloth/QLoRA.
10. Re-run baselines + custom eval on each fine-tune; compare summaries.

Publishable artifacts at the end: the corpus manifest with splits, the custom
`test.jsonl`, the contamination report, all `summary.json`/`scores.json` per
model, and the analytics report from generation.

## Part 6 — Still open (decisions for you, not code bugs)

- **HotpotQA**: not native to lm-eval. Options: (a) lean on the cross-page
  subset of the custom set for the multi-hop story, or (b) register a custom
  hotpotqa task YAML. Which do you want?
- **Judge model** for Tier 3 (`gpt-4o-mini` default vs a stronger judge).
- **LoRA rank/alpha and epochs** (`eval_plan.md` §7).
- **Generation cost**: confirm budget after the Step 5 dry-run extrapolation.
