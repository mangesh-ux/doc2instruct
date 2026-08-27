"""Regression tests for the citation matcher.

These encode the two defects that the 2026-08-27 smoke run found:

1. A faithful quote transcribed from a page *image* does not substring-match
   PyMuPDF-extracted text, because of hyphenated line breaks, ligatures and
   extra newlines. The old matcher rejected 31/40 local items for this.
2. Models elide material with "...". Each fragment is verbatim, but no single
   contiguous span matches. The old matcher rejected those too, even though
   every fragment was on the page.

A matcher that is too loose (accepting paraphrases) is as bad as one that is
too strict. The paraphrase cases below must stay rejected.
"""

from __future__ import annotations

import unittest

from multimodal_dataset.quality import (
    citation_coverage,
    has_citation_match,
    normalize_for_match,
    quote_coverage,
    quote_fragments,
)


HYPHENATED_PAGE = (
    "Transfer learning is usually studied as a consequence of distribu-\n"
    "tion shift. This paper identifies an orthogonal failure mode."
)

ELLIPSIS_PAGE = (
    "There are currently two interpretations of analog computing. "
    "One views analog and digital computing as opposites. "
    "A long intervening paragraph that is not part of the citation. "
    "The other emphasizes the analogy between physical systems and computation."
)


class NormalizeForMatchTests(unittest.TestCase):
    def test_rejoins_hyphenated_line_break(self) -> None:
        self.assertIn("distribution shift", normalize_for_match(HYPHENATED_PAGE))

    def test_folds_ligature(self) -> None:
        # NFKC folds the fi ligature so "fi" in a transcribed quote still hits.
        self.assertEqual(normalize_for_match("efﬁcient"), "efficient")

    def test_unifies_whitespace_and_case(self) -> None:
        self.assertEqual(normalize_for_match("Hello\n\tWORLD"), "hello world")


class QuoteCoverageTests(unittest.TestCase):
    def test_exact_containment_is_one(self) -> None:
        self.assertEqual(quote_coverage("abc def ghi", "def"), 1.0)

    def test_hyphenated_page_matches_transcribed_quote(self) -> None:
        quote = "consequence of distribution shift"
        self.assertGreaterEqual(quote_coverage(HYPHENATED_PAGE, quote), 0.85)

    def test_paraphrase_is_well_below_threshold(self) -> None:
        quote = "this work instead studies a completely different phenomenon"
        self.assertLess(quote_coverage(HYPHENATED_PAGE, quote), 0.5)


class EllipsisFragmentTests(unittest.TestCase):
    def test_splits_on_ellipsis(self) -> None:
        fragments = quote_fragments(
            "There are currently two interpretations of analog computing. "
            "One views analog and digital computing as opposites... "
            "The other emphasizes the analogy between physical systems and computation."
        )
        self.assertEqual(len(fragments), 2)

    def test_elided_quote_is_accepted_when_both_fragments_are_on_the_page(self) -> None:
        quote = (
            "There are currently two interpretations of analog computing. "
            "One views analog and digital computing as opposites... "
            "The other emphasizes the analogy between physical systems and computation."
        )
        self.assertGreaterEqual(citation_coverage(ELLIPSIS_PAGE, quote), 0.85)
        self.assertTrue(has_citation_match(ELLIPSIS_PAGE, quote))

    def test_elided_quote_rejected_when_second_fragment_is_invented(self) -> None:
        quote = (
            "There are currently two interpretations of analog computing... "
            "this second clause was never written anywhere on the page at all"
        )
        self.assertFalse(has_citation_match(ELLIPSIS_PAGE, quote))


class HasCitationMatchTests(unittest.TestCase):
    def test_empty_inputs_fail(self) -> None:
        self.assertFalse(has_citation_match("", "hello world"))
        self.assertFalse(has_citation_match("hello world", ""))

    def test_short_quote_fails(self) -> None:
        self.assertFalse(has_citation_match("abcdefghi jklmnop", "ab"))

    def test_newlines_in_page_do_not_block_single_line_quote(self) -> None:
        # This is the Stage-2 killer: pack text keeps PDF newlines, model quotes
        # do not. Bare `.lower()` substring matching can never succeed here.
        page = "Bayes sufficiency means that H contains\nthe Bayes quotient."
        quote = "Bayes sufficiency means that H contains the Bayes quotient."
        self.assertTrue(has_citation_match(page, quote))


if __name__ == "__main__":
    unittest.main()
