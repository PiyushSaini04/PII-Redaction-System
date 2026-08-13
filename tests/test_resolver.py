"""
tests/test_resolver.py
-----------------------
Tests for resolver.resolve_overlaps() and collect_all_matches().

Key invariants:
  - Regex beats NER on the same span
  - Longer span wins when both same detector type
  - Non-overlapping matches are all kept
  - All N occurrences of the same text are preserved (not collapsed)
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from models import TextBlock, PIIMatch
from resolver import resolve_overlaps


def block(text="test block"):
    return TextBlock(text=text, location={"type": "paragraph", "para_idx": 0,
                                          "runs": [], "run_offsets": []})


def make_match(text, cat, start, end, blk, detector="regex", confidence=1.0):
    return PIIMatch(
        text=text, category=cat,
        start_char=start, end_char=end,
        source_block=blk,
        confidence=confidence,
        detector=detector,
    )


class TestResolveOverlaps:
    def test_regex_beats_ner_on_same_span(self):
        b = block("john@example.com")
        regex_m = make_match("john@example.com", "EMAIL", 0, 16, b, detector="regex")
        ner_m = make_match("john@example.com", "PERSON", 0, 16, b, detector="ner")
        result = resolve_overlaps([ner_m, regex_m])
        assert len(result) == 1
        assert result[0].category == "EMAIL"
        assert result[0].detector == "regex"

    def test_non_overlapping_kept(self):
        b = block("foo@bar.com baz@qux.com")
        m1 = make_match("foo@bar.com", "EMAIL", 0, 11, b)
        m2 = make_match("baz@qux.com", "EMAIL", 12, 23, b)
        result = resolve_overlaps([m1, m2])
        assert len(result) == 2

    def test_longer_span_wins_same_detector(self):
        b = block("John Smith")
        short = make_match("John", "PERSON", 0, 4, b, detector="ner")
        long_ = make_match("John Smith", "PERSON", 0, 10, b, detector="ner")
        result = resolve_overlaps([short, long_])
        assert len(result) == 1
        assert result[0].text == "John Smith"

    def test_different_blocks_not_compared(self):
        b1 = block("text one")
        b2 = block("text two")
        m1 = make_match("text", "PERSON", 0, 4, b1, detector="ner")
        m2 = make_match("text", "PERSON", 0, 4, b2, detector="ner")
        result = resolve_overlaps([m1, m2])
        assert len(result) == 2  # different blocks — both kept

    def test_all_occurrences_preserved(self):
        """5 non-overlapping occurrences of the same name must all be kept."""
        b = block("A " * 50)  # large block
        matches = []
        for i in range(5):
            start = i * 10
            matches.append(make_match("A", "PERSON", start, start + 1, b, detector="ner"))
        result = resolve_overlaps(matches)
        assert len(result) == 5

    def test_empty_input(self):
        assert resolve_overlaps([]) == []

    def test_single_match_returned(self):
        b = block("hello")
        m = make_match("hello", "PERSON", 0, 5, b)
        result = resolve_overlaps([m])
        assert result == [m]

    def test_partial_overlap_regex_wins(self):
        """When regex and NER partially overlap, regex match is kept."""
        b = block("+91 9876543210 is a phone and also tagged as entity")
        regex_m = make_match("+91 9876543210", "PHONE", 0, 14, b, detector="regex")
        ner_m = make_match("9876543210 is", "PERSON", 4, 17, b, detector="ner")
        result = resolve_overlaps([regex_m, ner_m])
        assert any(m.detector == "regex" for m in result)
