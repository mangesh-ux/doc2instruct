# Dataset card — doc2instruct held-out evaluation set

`eval/custom/test.jsonl` — 150 grounded reading-comprehension questions over 15
arXiv papers that were **excluded from all training data**.

Its purpose is narrow: measure whether training on doc2instruct-generated data
improves grounded question answering on documents the model has never seen, and
in particular whether Stage 2 (cross-page synthesis) buys anything beyond
Stage 1 (single-page).

---

## 1. Composition

| | Count |
|---|---|
| Total items | 150 |
| `single_page` | 90 |
| `cross_page` (verified multi-hop) | 60 |
| cross/single ratio | 0.67 |
| Distinct source papers | 15 |
| Evidence exactly verbatim | 146 / 150 |
| Mean evidence coverage | 0.998 |

Source papers are arXiv `cs.CL` / `cs.LG` / `cs.AI`, 2023 onward, split from the
corpus by `scripts/split_corpus.py` with seed 42 and recorded in
`corpus/manifest.jsonl`.

## 2. Schema

```json
{
  "id": "2606.13178_p1-2_0000",
  "paper_id": "2606.13178",
  "arxiv_id": "2606.13178",
  "question_type": "single_page | cross_page",
  "pages": [1, 2],
  "question": "...",
  "answer": "...",
  "evidence_quote": "verbatim span supporting the answer",
  "evidence_quotes": [{"page": 1, "quote": "..."}, {"page": 2, "quote": "..."}],
  "why_both_pages": "which fact came from which page",
  "provenance": { "...": "see below" }
}
```

`evidence_quotes` and `why_both_pages` are present only on `cross_page` items.
`answer` is deliberately short (a phrase or one sentence) so exact-match and
token-F1 scoring are meaningful.

## 3. How it was built

1. **`bootstrap_eval_set.py`** — extract page text from the held-out PDFs and
   draft candidate questions.
2. **`improve_candidates.py`** — rewrite every candidate under a strict rubric
   (`gpt-4.1`): specific verifiable answer, no generic "main contribution"
   questions, no yes/no answers, evidence copied character-for-character.
3. **`build_cross_page_eval.py`** — regenerate cross-page items under a schema
   that demands **one verbatim span per page** plus a statement of which fact
   comes from which page.
4. **`ablate_cross_page.py`** — the multi-hop test (see §4).
5. **`verify_candidates.py`** — mechanical gate, then quota-balanced selection.
6. **`finalize_eval_set.py`** — schema and provenance.

Every step is re-runnable and writes a JSON report:
`verification_report.json`, `ablation_report.json`, `test_provenance.json`.

## 4. What "verified multi-hop" means

Asking a model to write a question that "requires both pages" does not produce
one. Measured here: of 136 items the generator produced *and self-certified* as
multi-hop, **82 (60.3%) were answerable from a single page**.

So multi-hop status is tested rather than asserted. For each item the judge
model answers the question given **only page A**, then **only page B**. Token F1
against the gold answer is recorded for both. An item survives only if *neither*
single page reaches F1 ≥ 0.6 — the same shortcut audit applied to multi-hop QA
benchmarks. The two scores are kept per item in
`provenance.single_page_f1`, so any item can be re-checked.

Across two drafting pools (136 + 207 self-certified items) the shortcut rate
was 60–63%. 73 items passed both the ablation and per-page verbatim grounding;
60 of them (round-robin across papers) form the published `cross_page` subset.

## 5. Verification each item passed

- **Evidence grounding** — the evidence span occurs in the source page text,
  normalized for the ways PDF extraction differs from a faithful transcription
  (unicode ligatures, hyphenated line breaks, whitespace). Coverage ≥ 0.85; 146
  of 150 are exact.
- **Per-page grounding** (cross-page) — each of the two spans is verbatim on
  *its own* page, so one page cannot supply all the evidence.
- **Multi-hop necessity** (cross-page) — the ablation in §4.
- **Question specificity** — generic templates rejected.
- **Answer scoreability** — short enough for EM/F1, never yes/no.
- **Deduplication** — near-duplicate questions removed (0.88 similarity).
- **Paper balance** — items selected round-robin across papers, so no paper
  dominates.

## 6. Limitations — read before quoting numbers

- **Not human-verified.** Every item passed the mechanical gate above, and
  `provenance.review` says `machine_gate` for all 150. Nobody has yet read them
  and signed off. Use `eval/review_candidates.ipynb` to spot-check; items you
  edit are re-stamped `human`. **Do not describe this set as hand-verified until
  that field says so.**
- **Questions are model-drafted (`gpt-4.1`).** Grounding is verified against the
  papers, and the papers were never trained on, but the *phrasing* comes from a
  model. If the model under test shares a lineage with the drafter, some
  stylistic affinity cannot be ruled out. The single-page ablation and verbatim
  checks constrain what that affinity can buy.
- **cross/single ratio is 0.67**, above the 0.5 floor. Getting there required
  a second drafting pool: the first 136 self-certified multi-hop items yielded
  only 35 after ablation + per-page grounding. A wider enumeration of page
  pairs (207 more drafts) brought the verified pool to 73, of which 60 were
  selected. The ablation still discards ~62% of drafted cross-page items; that
  is a property of the task, not a bug.
- **Answers are short by construction**, so this set measures precise factual
  grounding, not long-form explanation quality.
- **15 papers is a small sample.** Per-paper variance will be visible; report
  confidence intervals, and prefer paired comparisons across model variants on
  identical items.
- **Domain is narrow** (recent arXiv ML/NLP). Nothing here speaks to other
  document types.

## 7. Intended use

Comparative evaluation of the three variants in `eval_plan.md` §3 (base,
+Stage 1, +Stage 1&2) on identical items. Report `single_page` and `cross_page`
subsets **separately** — the cross-page subset is the only part that tests the
Stage 2 claim. With 60 items it is still the noisier of the two; report an
interval, not just a point.

Not intended as a general-purpose benchmark or leaderboard.

## 8. Contamination control

`scripts/check_contamination.py` enforces two independent checks against
`corpus/manifest.jsonl`: no held-out paper identifier may appear in the training
data, and no held-out page text may overlap it above a shingled k-gram
threshold. It must exit 0 before fine-tuning. `eval/finetune_qlora.ipynb`
re-checks at the point of use.

## 9. Reproduction

```bash
python scripts/bootstrap_eval_set.py --use-openai
python scripts/improve_candidates.py --workers 8 --out-csv eval/custom/candidates_v3.csv
python scripts/build_cross_page_eval.py --enumerate-pairs --workers 8
python scripts/ablate_cross_page.py --workers 8
python scripts/verify_candidates.py \
  --in-csv eval/custom/cross_page_ablated.csv \
           eval/custom/cross_page_ablated_wide.csv \
           eval/custom/candidates_v3.csv \
  --target-single 90 --target-cross 60 --require-ablation-for-cross
python scripts/finalize_eval_set.py
```

Item selection is deterministic given the same candidate CSVs. The drafting and
ablation steps call an LLM and so are not bit-reproducible; the per-item
provenance and the JSON reports are what make a given build auditable.
