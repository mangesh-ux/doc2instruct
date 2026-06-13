"""Helpers for the custom HotpotQA lm-evaluation-harness task.

HotpotQA is not a native lm-eval task, but it is the headline benchmark for the
Stage-2 cross-page / multi-hop hypothesis, so we evaluate on the *real* dataset
(distractor setting) rather than a proxy. This module provides:

- doc_to_text:      flatten the 10 distractor paragraphs + question into a prompt
- doc_to_target:    the gold short answer
- process_results:  SQuAD-style normalized Exact Match and token F1 on the answer

Answer-only EM/F1 (with SQuAD normalization) is the standard, comparable way to
report HotpotQA answer quality for generative models. We do not score the
supporting-facts sub-metric (it needs sentence-id supervision the model can't
emit here); answer EM/F1 is the credible, apples-to-apples number.
"""

from __future__ import annotations

import re
import string
from collections import Counter


def _normalize(text: str) -> str:
    """SQuAD/HotpotQA answer normalization: lowercase, strip punctuation/articles."""
    text = text.lower()
    text = "".join(ch for ch in text if ch not in set(string.punctuation))
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def doc_to_text(doc) -> str:
    titles = doc["context"]["title"]
    sentences = doc["context"]["sentences"]
    paragraphs = []
    for title, sents in zip(titles, sentences):
        paragraphs.append(f"{title}: {' '.join(sents)}")
    context = "\n".join(paragraphs)
    return (
        "Answer the multi-hop question using only the given context. "
        "Reply with the short answer span only.\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {doc['question']}\n"
        "Answer:"
    )


def doc_to_target(doc) -> str:
    return " " + doc["answer"]


def _token_f1(pred: str, gold: str) -> float:
    p_tokens = _normalize(pred).split()
    g_tokens = _normalize(gold).split()
    if not p_tokens or not g_tokens:
        return float(p_tokens == g_tokens)
    common = Counter(p_tokens) & Counter(g_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(p_tokens)
    recall = num_same / len(g_tokens)
    return 2 * precision * recall / (precision + recall)


def process_results(doc, results) -> dict:
    # Greedy generation; take the first line as the answer span.
    prediction = (results[0] or "").strip().split("\n")[0].strip()
    gold = doc["answer"]
    return {
        "exact_match": float(_normalize(prediction) == _normalize(gold)),
        "f1": _token_f1(prediction, gold),
    }
