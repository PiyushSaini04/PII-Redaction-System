"""
tests/test_evaluate.py
-----------------------
Tests for evaluate.py: scoring, metric computation, and report generation.

All tests use synthetic ground-truth / detection pairs with hand-computed
expected metrics so results can be verified independently.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import json
import math
import tempfile
from pathlib import Path

import pytest
from evaluate import score, generate_report, load_ground_truth, load_detections


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _det(text, category, para_idx=0, start_char=0):
    return {
        "text": text,
        "category": category,
        "start_char": start_char,
        "end_char": start_char + len(text),
        "source_location": {"type": "paragraph", "para_idx": para_idx},
        "confidence": 1.0,
        "needs_review": False,
        "detector": "regex",
    }


def _gt(text, category, para_idx=0, start_char=None):
    rec = {"text": text, "category": category, "para_idx": para_idx}
    if start_char is not None:
        rec["start_char"] = start_char
        rec["end_char"] = start_char + len(text)
    return rec


# ---------------------------------------------------------------------------
# score()
# ---------------------------------------------------------------------------

class TestScore:
    def test_perfect_detection(self):
        gt = [_gt("john@example.com", "EMAIL", para_idx=0)]
        dets = [_det("john@example.com", "EMAIL", para_idx=0)]
        result = score(dets, gt)
        assert result["EMAIL"]["TP"] == 1
        assert result["EMAIL"]["FP"] == 0
        assert result["EMAIL"]["FN"] == 0
        assert math.isclose(result["EMAIL"]["precision"], 1.0)
        assert math.isclose(result["EMAIL"]["recall"], 1.0)

    def test_false_positive(self):
        gt = []
        dets = [_det("fake@nogt.com", "EMAIL")]
        result = score(dets, gt)
        assert result["EMAIL"]["TP"] == 0
        assert result["EMAIL"]["FP"] == 1
        assert result["EMAIL"]["FN"] == 0
        assert result["EMAIL"]["precision"] == 0.0

    def test_false_negative(self):
        gt = [_gt("missed@example.com", "EMAIL")]
        dets = []
        result = score(dets, gt)
        assert result["EMAIL"]["TP"] == 0
        assert result["EMAIL"]["FP"] == 0
        assert result["EMAIL"]["FN"] == 1
        assert result["EMAIL"]["recall"] == 0.0

    def test_precision_and_recall_formula(self):
        # 2 TP, 1 FP, 1 FN
        gt = [
            _gt("alice@x.com", "EMAIL", para_idx=0),
            _gt("bob@x.com",   "EMAIL", para_idx=1),
        ]
        dets = [
            _det("alice@x.com",  "EMAIL", para_idx=0),  # TP
            _det("bob@x.com",    "EMAIL", para_idx=1),  # TP
            _det("wrong@x.com",  "EMAIL", para_idx=2),  # FP
        ]
        result = score(dets, gt)
        assert result["EMAIL"]["TP"] == 2
        assert result["EMAIL"]["FP"] == 1
        assert result["EMAIL"]["FN"] == 0
        assert math.isclose(result["EMAIL"]["precision"], 2/3, rel_tol=1e-6)
        assert math.isclose(result["EMAIL"]["recall"], 1.0, rel_tol=1e-6)

    def test_category_with_no_gt_instances(self):
        """Categories with zero GT instances must be present with N/A metrics."""
        gt = [_gt("john@test.com", "EMAIL")]
        dets = [_det("john@test.com", "EMAIL")]
        result = score(dets, gt)
        # SSN has no GT or dets
        assert result["SSN"]["TP"] == 0
        assert result["SSN"]["FP"] == 0
        assert result["SSN"]["FN"] == 0
        assert result["SSN"]["precision"] is None
        assert result["SSN"]["recall"] is None

    def test_overall_aggregate(self):
        gt = [
            _gt("alice@x.com", "EMAIL"),
            _gt("John Smith",  "PERSON"),
        ]
        dets = [
            _det("alice@x.com", "EMAIL"),   # TP EMAIL
            _det("John Smith",  "PERSON"),  # TP PERSON
        ]
        result = score(dets, gt)
        assert result["OVERALL"]["TP"] == 2
        assert result["OVERALL"]["FP"] == 0
        assert result["OVERALL"]["FN"] == 0
        assert math.isclose(result["OVERALL"]["precision"], 1.0)
        assert math.isclose(result["OVERALL"]["recall"], 1.0)

    def test_fp_examples_populated(self):
        gt = []
        dets = [_det("spurious@fp.com", "EMAIL")]
        result = score(dets, gt)
        assert len(result["EMAIL"]["fp_examples"]) == 1

    def test_fn_examples_populated(self):
        gt = [_gt("missed@example.com", "EMAIL")]
        dets = []
        result = score(dets, gt)
        assert len(result["EMAIL"]["fn_examples"]) == 1


# ---------------------------------------------------------------------------
# generate_report()
# ---------------------------------------------------------------------------

class TestGenerateReport:
    def _run_report(self, gt, dets):
        scores = score(dets, gt)
        with tempfile.TemporaryDirectory() as tmpdir:
            report_path = Path(tmpdir) / "report.md"
            generate_report(scores, report_path)
            return report_path.read_text(encoding="utf-8")

    def test_report_contains_overall_section(self):
        gt = [_gt("john@test.com", "EMAIL")]
        dets = [_det("john@test.com", "EMAIL")]
        md = self._run_report(gt, dets)
        assert "Overall Summary" in md

    def test_report_contains_per_category_section(self):
        gt = [_gt("john@test.com", "EMAIL")]
        dets = [_det("john@test.com", "EMAIL")]
        md = self._run_report(gt, dets)
        assert "Per-Category Results" in md

    def test_all_nine_categories_present(self):
        gt = [_gt("john@test.com", "EMAIL")]
        dets = [_det("john@test.com", "EMAIL")]
        md = self._run_report(gt, dets)
        for cat in ["PERSON", "EMAIL", "PHONE", "COMPANY", "ADDRESS",
                    "SSN", "CREDIT_CARD", "DOB", "IP"]:
            assert cat in md, f"Category {cat} missing from report"

    def test_fp_appendix_present(self):
        gt = []
        dets = [_det("bad@fp.com", "EMAIL")]
        md = self._run_report(gt, dets)
        assert "False Positive" in md
        assert "bad@fp.com" in md

    def test_fn_appendix_present(self):
        gt = [_gt("missed@example.com", "EMAIL")]
        dets = []
        md = self._run_report(gt, dets)
        assert "False Negative" in md
        assert "missed@example.com" in md

    def test_absent_category_shows_na(self):
        """Categories with no GT must show N/A, not be hidden."""
        gt = [_gt("john@test.com", "EMAIL")]
        dets = [_det("john@test.com", "EMAIL")]
        md = self._run_report(gt, dets)
        # SSN has no GT — should show N/A
        assert "N/A" in md


# ---------------------------------------------------------------------------
# load_ground_truth() validation
# ---------------------------------------------------------------------------

class TestLoadGroundTruth:
    def test_valid_file(self, tmp_path):
        data = [{"text": "John", "category": "PERSON"}]
        p = tmp_path / "gt.json"
        p.write_text(json.dumps(data), encoding="utf-8")
        result = load_ground_truth(str(p))
        assert result == data

    def test_missing_text_field_raises(self, tmp_path):
        data = [{"category": "PERSON"}]
        p = tmp_path / "gt.json"
        p.write_text(json.dumps(data), encoding="utf-8")
        with pytest.raises(ValueError, match="missing"):
            load_ground_truth(str(p))

    def test_not_a_list_raises(self, tmp_path):
        p = tmp_path / "gt.json"
        p.write_text(json.dumps({"text": "John", "category": "PERSON"}),
                     encoding="utf-8")
        with pytest.raises(ValueError):
            load_ground_truth(str(p))
