# doc2instruct — evaluation results

> **Status: template. No numbers below have been produced yet.**
> Every cell reads `TBD` until the corresponding run finishes. Fill cells in
> only from a completed run, and paste the run's output path next to it.
> Do not delete rows that turn out unfavourable — a Stage 2 that does not help
> is the single most useful thing this evaluation can tell you.

## 0. What is being compared

| Variant | Training data | Adapter |
|---|---|---|
| **A. Base** | none | — |
| **B. +Stage 1** | `output/train_run/chatml_dataset.jsonl` | `outputs/doc2instruct-stage1/adapter` |
| **C. +Stage 1&2** | Stage 1 + `cross_page_chatml_dataset.jsonl` | `outputs/doc2instruct-stage1_stage2/adapter` |

Base model: `Qwen/Qwen2.5-7B` (base, not Instruct). QLoRA, r=16, alpha=32,
2 epochs, seed 42 — see each run's `run_manifest.json` for the exact values and
dataset SHA256s.

**Preconditions for these numbers to mean anything**
- [ ] `scripts/check_contamination.py` exited 0 against the training dataset
- [ ] All three variants evaluated on the *identical* `eval/custom/test.jsonl`
- [ ] Custom eval set spot-checked by a human (see `DATASET_CARD.md` §6)

---

## 1. Tier 1 — no-regression sanity

Does instruction-tuning on document QA damage general capability? Large drops
here mean the gains below are bought with catastrophic forgetting.

| Task | Metric | A. Base | B. +Stage 1 | C. +Stage 1&2 |
|---|---|---|---|---|
| MMLU | acc | TBD | TBD | TBD |
| GSM8K (CoT) | exact_match | TBD | TBD | TBD |

Command: `python scripts/run_baselines.py --model-args pretrained=<...> --output-dir eval/baselines/<variant>`

## 2. Tier 2 — document QA benchmarks

| Task | Metric | A. Base | B. +Stage 1 | C. +Stage 1&2 |
|---|---|---|---|---|
| SQuAD 2.0 | F1 | TBD | TBD | TBD |
| Qasper | F1 | TBD | TBD | TBD |
| HotpotQA (distractor) | EM | TBD | TBD | TBD |
| HotpotQA (distractor) | F1 | TBD | TBD | TBD |

HotpotQA is the external check on the Stage 2 claim: it is real, human-authored
multi-hop QA that nothing in this project produced. If Stage 2 helps on our own
cross-page subset but not here, say so plainly.

## 3. Tier 3 — custom held-out set

`eval/custom/test.jsonl` (125 items over 15 held-out papers).
**Report the subsets separately.** The cross-page subset is the only part that
tests Stage 2, and at 35 items it is noisy.

### Overall

| Metric | A. Base | B. +Stage 1 | C. +Stage 1&2 |
|---|---|---|---|
| Exact match | TBD | TBD | TBD |
| Token F1 | TBD | TBD | TBD |
| LLM-judge correct % | TBD | TBD | TBD |

### `single_page` subset (n=90) — tests Stage 1

| Metric | A. Base | B. +Stage 1 | C. +Stage 1&2 |
|---|---|---|---|
| Exact match | TBD | TBD | TBD |
| Token F1 | TBD | TBD | TBD |

### `cross_page` subset (n=35) — tests Stage 2

| Metric | A. Base | B. +Stage 1 | C. +Stage 1&2 |
|---|---|---|---|
| Exact match | TBD | TBD | TBD |
| Token F1 | TBD | TBD | TBD |

Command: `python scripts/run_custom_eval.py --backend hf --model <...> --out eval/custom/results/<variant>.json`

Judge model: `gpt-4.1` (independent of the EM/F1 path).

---

## 4. Reading these numbers honestly

With n=90 and n=35, differences of a few points are noise. Before claiming an
effect:

- **Use paired comparisons.** Same items, same prompts; report how many items
  flipped correct→incorrect and incorrect→correct, not just the aggregate.
- **Give an interval.** A bootstrap CI over items is enough; on n=35 the 95% CI
  is roughly ±16 points for a proportion near 50%. Any cross-page claim smaller
  than that is not supported.
- **Separate the two claims.** "Stage 1 helps" (B vs A) and "Stage 2 adds
  something" (C vs B) are different questions with different evidence.
- **Check Tier 1 first.** A gain on document QA alongside a large MMLU/GSM8K
  drop is a trade, not an improvement.
- **Cross-check Stage 2 against HotpotQA.** Our cross-page subset is
  model-drafted from the same pipeline family as the training data; HotpotQA is
  not. Agreement between them is what makes the claim credible.

## 5. Known caveats to carry into any write-up

- Custom eval questions are model-drafted, then mechanically verified
  (verbatim evidence, single-page ablation). See `DATASET_CARD.md` §6.
- cross/single ratio is 0.39, below the 0.5 target, because the multi-hop
  ablation rejects ~74% of drafted cross-page candidates.
- 15 held-out papers, single domain (recent arXiv ML/NLP).
- Stage 1 and Stage 2 record counts are unequal; if Stage 2 adds far fewer
  records, C vs B also differs in dataset *size*. Note the counts from
  `run_manifest.json` so a size effect isn't mistaken for a Stage 2 effect.

## 6. Run log

| Date | Variant | Step | Output path | Notes |
|---|---|---|---|---|
| | | | | |
