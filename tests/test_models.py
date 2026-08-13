"""
tests/test_models.py
--------------------
Tests for TextBlock and PIIMatch dataclasses (models.py).
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from models import TextBlock, PIIMatch


def _make_block(text="Hello world"):
    return TextBlock(
        text=text,
        location={"type": "paragraph", "para_idx": 0, "runs": [], "run_offsets": []},
    )


class TestTextBlock:
    def test_fields_stored(self):
        b = _make_block("test text")
        assert b.text == "test text"
        assert b.location["type"] == "paragraph"
        assert b.location["para_idx"] == 0

    def test_location_carries_runs(self):
        b = _make_block()
        assert "runs" in b.location
        assert "run_offsets" in b.location


class TestPIIMatch:
    def test_default_confidence_is_one(self):
        b = _make_block()
        m = PIIMatch(text="John", category="PERSON",
                     start_char=0, end_char=4, source_block=b)
        assert m.confidence == 1.0
        assert m.needs_review is False
        assert m.detector == ""

    def test_custom_fields(self):
        b = _make_block()
        m = PIIMatch(text="addr", category="ADDRESS",
                     start_char=5, end_char=9, source_block=b,
                     confidence=0.75, needs_review=True, detector="heuristic")
        assert m.confidence == 0.75
        assert m.needs_review is True
        assert m.detector == "heuristic"

    def test_source_block_reference(self):
        b = _make_block("the quick brown fox")
        m = PIIMatch(text="quick", category="PERSON",
                     start_char=4, end_char=9, source_block=b)
        assert m.source_block is b
        assert b.text[m.start_char:m.end_char] == "quick"
