"""
evaluate.py
-----------
Evaluation module: scores detector output against a human-verified ground
truth set and generates evaluation_report.md.

Evaluation methodology — span/entity-level
------------------------------------------
A detection is a True Positive (TP) if there is a ground-truth record with:
  - Same category (case-insensitive)  AND
  - Matching text (normalised whitespace)  AND
  - Overlapping or matching source location (para_idx ± tolerance, or
    exact text match when location is unavailable)

A detection with no matching ground-truth record -> False Positive (FP)
A ground-truth record with no matching detection -> False Negative (FN)

True Negatives (TN) are not computable at span level without exhaustive
annotation of every non-PII token; character-level accuracy is therefore
NOT reported as a primary metric.

Primary metrics (per category and overall):
  Precision = TP / (TP + FP)
  Recall    = TP / (TP + FN)
  F1        = 2 × P × R / (P + R)

Internal quality targets (not assignment requirements):
  Precision ≥ 0.80,  Recall ≥ 0.85

Usage
-----
python evaluate.py \
    --detections data/output/detections.json \
    --ground-truth ground_truth.json \
    --report data/output/evaluation_report.md
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Optional

# All 9 required categories
ALL_CATEGORIES = [
    "PERSON", "EMAIL", "PHONE", "COMPANY",
    "ADDRESS", "SSN", "CREDIT_CARD", "DOB", "IP",
]


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

def load_ground_truth(path: str | Path) -> list[dict]:
    """
    Load and minimally validate ground_truth.json.

    Expected schema per record (all fields optional except text & category):
    {
        "text":      str,            # exact PII string
        "category":  str,            # one of the 9 required labels
        "para_idx":  int | null,     # source paragraph index (if known)
        "start_char": int | null,
        "end_char":   int | null,
        "notes":      str | null
    }

    Raises ValueError on schema violations.
    """
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("ground_truth.json must be a JSON array.")
    for i, rec in enumerate(data):
        if "text" not in rec or "category" not in rec:
            raise ValueError(
                f"ground_truth.json record {i} missing 'text' or 'category'."
            )
    return data


def load_detections(path: str | Path) -> list[dict]:
    """
    Load detections.json (output of redact.py --dry-run or main run).

    Expected schema per record:
    {
        "text":           str,
        "category":       str,
        "start_char":     int,
        "end_char":       int,
        "source_location": dict,
        "confidence":     float,
        "needs_review":   bool,
        "detector":       str
    }
    """
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("detections.json must be a JSON array.")
    return data


# ---------------------------------------------------------------------------
# Matching logic
# ---------------------------------------------------------------------------

def _normalise(text: str) -> str:
    """Normalise whitespace for comparison."""
    return " ".join(text.split()).lower()


def _texts_match(det_text: str, gt_text: str) -> bool:
    return _normalise(det_text) == _normalise(gt_text)


def _location_matches(det: dict, gt: dict, char_tolerance: int = 10) -> bool:
    """
    Check whether a detection and a ground-truth record refer to the same
    location in the document.

    Tries (in order):
      1. para_idx matches exactly (if both have it)
      2. start_char is within ±char_tolerance of gt start_char
      3. Text-only match (location not available in one or both)
    """
    det_loc = det.get("source_location", {})
    gt_para = gt.get("para_idx")
    det_para = det_loc.get("para_idx")

    # Both have para_idx
    if gt_para is not None and det_para is not None:
        return gt_para == det_para

    # Both have start_char
    gt_start = gt.get("start_char")
    det_start = det.get("start_char")
    if gt_start is not None and det_start is not None:
        return abs(gt_start - det_start) <= char_tolerance

    # Fall back to text match only (allow if text is unique)
    return True


def _is_match(det: dict, gt: dict) -> bool:
    """Return True if *det* and *gt* represent the same PII occurrence."""
    return (
        det["category"].upper() == gt["category"].upper()
        and _texts_match(det["text"], gt["text"])
        and _location_matches(det, gt)
    )


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def score(
    detections: list[dict],
    ground_truth: list[dict],
) -> dict:
    """
    Compute per-category and overall TP/FP/FN/Precision/Recall/F1.

    Returns
    -------
    dict keyed by category label + "OVERALL", each value is:
    {
        "TP": int, "FP": int, "FN": int,
        "precision": float | None,
        "recall": float | None,
        "f1": float | None,
        "fp_examples": list[dict],   # detections with no GT match
        "fn_examples": list[dict],   # GT records with no detection match
    }
    """
    results: dict[str, dict] = {}

    for cat in ALL_CATEGORIES:
        cat_dets = [d for d in detections if d["category"].upper() == cat]
        cat_gt = [g for g in ground_truth if g["category"].upper() == cat]

        gt_matched = [False] * len(cat_gt)
        det_matched = [False] * len(cat_dets)
        tp = 0

        for di, det in enumerate(cat_dets):
            for gi, gt in enumerate(cat_gt):
                if not gt_matched[gi] and _is_match(det, gt):
                    tp += 1
                    det_matched[di] = True
                    gt_matched[gi] = True
                    break

        fp = sum(1 for m in det_matched if not m)
        fn = sum(1 for m in gt_matched if not m)

        precision = tp / (tp + fp) if (tp + fp) > 0 else None
        recall = tp / (tp + fn) if (tp + fn) > 0 else None
        f1 = (
            2 * precision * recall / (precision + recall)
            if precision and recall
            else None
        )

        fp_examples = [cat_dets[i] for i, m in enumerate(det_matched) if not m]
        fn_examples = [cat_gt[i] for i, m in enumerate(gt_matched) if not m]

        results[cat] = {
            "TP": tp, "FP": fp, "FN": fn,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "fp_examples": fp_examples,
            "fn_examples": fn_examples,
        }

    # Overall aggregate
    total_tp = sum(results[c]["TP"] for c in ALL_CATEGORIES)
    total_fp = sum(results[c]["FP"] for c in ALL_CATEGORIES)
    total_fn = sum(results[c]["FN"] for c in ALL_CATEGORIES)
    overall_p = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else None
    overall_r = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else None
    overall_f1 = (
        2 * overall_p * overall_r / (overall_p + overall_r)
        if overall_p and overall_r else None
    )
    results["OVERALL"] = {
        "TP": total_tp, "FP": total_fp, "FN": total_fn,
        "precision": overall_p,
        "recall": overall_r,
        "f1": overall_f1,
        "fp_examples": [],
        "fn_examples": [],
    }

    return results


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------

def _fmt(value: Optional[float], na: str = "N/A") -> str:
    if value is None:
        return na
    return f"{value:.3f}"


def _target_flag(value: Optional[float], threshold: float) -> str:
    """Return ✓ or ✗ against an internal quality target."""
    if value is None:
        return "—"
    return "✓" if value >= threshold else "✗"


def generate_report(
    scores: dict,
    output_path: str | Path,
) -> None:
    """
    Write evaluation_report.md with:
      - Overall summary table
      - Per-category results table (TP / FP / FN / P / R / F1)
      - False Positive appendix
      - False Negative appendix

    Categories with zero ground-truth instances are listed with N/A metrics
    and an explanatory note — they are not hidden.
    Internal quality targets are shown as advisory flags, not requirements.
    """
    lines: list[str] = []

    def w(line: str = "") -> None:
        lines.append(line)

    w("# PII Redaction Tool — Evaluation Report")
    w()
    w("> **Evaluation methodology:** span/entity-level matching.")
    w("> A detection is a True Positive (TP) when category, text (normalised),")
    w("> and source location all agree with a ground-truth record.")
    w("> Character-level accuracy is not reported (TN cannot be computed at span level).")
    w("> Internal quality targets: Precision ≥ 0.80, Recall ≥ 0.85 (not assignment requirements).")
    w()

    # --- Overall summary ---
    w("## Overall Summary")
    w()
    ov = scores["OVERALL"]
    w(f"| Metric | Value |")
    w(f"|--------|-------|")
    w(f"| True Positives (TP) | {ov['TP']} |")
    w(f"| False Positives (FP) | {ov['FP']} |")
    w(f"| False Negatives (FN) | {ov['FN']} |")
    w(f"| **Precision** | **{_fmt(ov['precision'])}** |")
    w(f"| **Recall** | **{_fmt(ov['recall'])}** |")
    w(f"| **F1** | **{_fmt(ov['f1'])}** |")
    w()

    # --- Per-category table ---
    w("## Per-Category Results")
    w()
    w("*P≥0.80* and *R≥0.85* are internal targets. "
      "N/A = no ground-truth instances in the source document.")
    w()
    w("| Category | TP | FP | FN | Precision | P≥0.80 | Recall | R≥0.85 | F1 |")
    w("|---|---|---|---|---|---|---|---|---|")

    for cat in ALL_CATEGORIES:
        s = scores[cat]
        has_gt = (s["TP"] + s["FN"]) > 0
        note = "" if has_gt else " *(no GT instances)*"
        w(
            f"| {cat}{note} "
            f"| {s['TP']} | {s['FP']} | {s['FN']} "
            f"| {_fmt(s['precision'])} | {_target_flag(s['precision'], 0.80)} "
            f"| {_fmt(s['recall'])} | {_target_flag(s['recall'], 0.85)} "
            f"| {_fmt(s['f1'])} |"
        )

    w()

    # --- False Positive appendix ---
    w("## Appendix A — False Positives")
    w()
    w("Detections that had no matching ground-truth record.")
    w("These represent over-redaction — legitimate non-PII text that was incorrectly flagged.")
    w()

    any_fp = False
    for cat in ALL_CATEGORIES:
        fps = scores[cat]["fp_examples"]
        if fps:
            any_fp = True
            w(f"### {cat}")
            w()
            w("| Text | Confidence | Detector | Source Location |")
            w("|---|---|---|---|")
            for fp in fps:
                loc = fp.get("source_location", {})
                loc_str = ", ".join(f"{k}={v}" for k, v in loc.items()
                                    if k != "run_offsets")
                w(
                    f"| `{fp['text']}` "
                    f"| {fp.get('confidence', '—')} "
                    f"| {fp.get('detector', '—')} "
                    f"| {loc_str} |"
                )
            w()

    if not any_fp:
        w("*No false positives detected.*")
        w()

    # --- False Negative appendix ---
    w("## Appendix B — False Negatives")
    w()
    w("Ground-truth records that were not matched by any detection.")
    w("These represent under-redaction — real PII that the tool missed.")
    w()

    any_fn = False
    for cat in ALL_CATEGORIES:
        fns = scores[cat]["fn_examples"]
        if fns:
            any_fn = True
            w(f"### {cat}")
            w()
            w("| Text | Notes |")
            w("|---|---|")
            for fn in fns:
                notes = fn.get("notes", "")
                w(f"| `{fn['text']}` | {notes} |")
            w()

    if not any_fn:
        w("*No false negatives — all ground-truth PII instances were detected.*")
        w()

    # Write file
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"  evaluation_report.md -> {output_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="evaluate",
        description="Score detector output against human-verified ground truth.",
    )
    parser.add_argument(
        "--detections", required=True,
        help="Path to detections.json (output of redact.py).",
    )
    parser.add_argument(
        "--ground-truth", required=True,
        help="Path to human-verified ground_truth.json.",
    )
    parser.add_argument(
        "--report",
        default="data/output/evaluation_report.md",
        help="Output path for evaluation_report.md.",
    )
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)

    try:
        print(f"Loading ground truth: {args.ground_truth}")
        gt = load_ground_truth(args.ground_truth)
        print(f"  {len(gt)} ground-truth record(s)")

        print(f"Loading detections: {args.detections}")
        dets = load_detections(args.detections)
        print(f"  {len(dets)} detection(s)")

        print("Scoring …")
        scores = score(dets, gt)

        ov = scores["OVERALL"]
        print(f"  Overall -> P={_fmt(ov['precision'])}  "
              f"R={_fmt(ov['recall'])}  F1={_fmt(ov['f1'])}")

        generate_report(scores, args.report)
        print("Done.")
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise

    return 0


if __name__ == "__main__":
    sys.exit(main())
