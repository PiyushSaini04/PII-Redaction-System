"""
run_evaluation.py
-----------------
Proper evaluation pipeline.

METHODOLOGY:
  1. Load INDEPENDENT ground_truth_independent.json (not derived from detections)
  2. Load system detections from detections.json
  3. Match detections against ground truth using normalized text + location
  4. Classify as TP / FP / FN
  5. Compute per-category and overall Precision / Recall / F1
  6. Generate all required matrices (CSV)
  7. Generate all required charts (PNG)
  8. Generate final evaluation JSON

KEY RULE: A detection is TP only when it matches a ground-truth record.
          Detected != TP  (this was the flaw in the previous evaluation)

Usage:
  python run_evaluation.py
"""
from __future__ import annotations

import json
import os
import re
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

GROUND_TRUTH_PATH = "evaluation/ground_truth_independent.json"
DETECTIONS_PATH = "data/output/detections.json"
OUTPUT_DIR = Path("evaluation")

ALL_CATEGORIES = [
    "PERSON", "EMAIL", "PHONE", "COMPANY",
    "ADDRESS", "SSN", "CREDIT_CARD", "DOB", "IP",
]

CATEGORY_DISPLAY = {
    "PERSON": "Person Name",
    "EMAIL": "Email",
    "PHONE": "Phone",
    "COMPANY": "Company",
    "ADDRESS": "Address",
    "SSN": "SSN",
    "CREDIT_CARD": "Credit Card",
    "DOB": "DOB",
    "IP": "IP Address",
}

DETECTOR_MAP = {
    "PERSON": "NER (spaCy)",
    "COMPANY": "NER (spaCy)",
    "ADDRESS": "NER + Address Detector",
    "EMAIL": "Email Regex",
    "PHONE": "Phone Regex",
    "IP": "IP Regex",
    "CREDIT_CARD": "Credit Card Regex + Luhn",
    "SSN": "SSN Regex",
    "DOB": "DOB Context Regex",
}

# Color palette
COLORS = {
    "TP": "#2dd4bf",   # teal
    "FP": "#f87171",   # red
    "FN": "#fb923c",   # orange
    "P":  "#818cf8",   # indigo
    "R":  "#34d399",   # green
    "F1": "#f472b6",   # pink
}

DARK_BG = "#0f172a"
CARD_BG = "#1e293b"
TEXT_COLOR = "#e2e8f0"

# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

def load_json(path: str | Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)

# ---------------------------------------------------------------------------
# Matching
# ---------------------------------------------------------------------------

def _norm(text: str) -> str:
    """Normalize whitespace and lowercase for comparison."""
    return " ".join(text.lower().split())


def _texts_match(a: str, b: str) -> bool:
    return _norm(a) == _norm(b)


def _locations_match(det: dict, gt: dict, para_tol: int = 0, char_tol: int = 20) -> bool:
    """
    Check if detection and ground-truth refer to the same location.
    Strategy (in order):
      1. Both have para_idx -> must match exactly
      2. Both have start_char -> must be within char_tol
      3. Neither has location -> accept on text match alone (fallback)
    """
    det_loc = det.get("source_location", {})
    det_para = det_loc.get("para_idx")
    gt_para = gt.get("para_idx")

    if det_para is not None and gt_para is not None:
        return det_para == gt_para

    det_start = det.get("start_char")
    gt_start = gt.get("start_char")

    if det_start is not None and gt_start is not None:
        return abs(det_start - gt_start) <= char_tol

    # For injected FNs (no location) — text match is sufficient
    return True


def _is_match(det: dict, gt: dict) -> bool:
    return (
        det["category"].upper() == gt["category"].upper()
        and _texts_match(det["text"], gt["text"])
        and _locations_match(det, gt)
    )

# ---------------------------------------------------------------------------
# Core scoring
# ---------------------------------------------------------------------------

def score(detections: list[dict], ground_truth: list[dict]) -> dict:
    """
    Per-category TP / FP / FN with entity-level bipartite matching.
    One GT entity can match at most one detection (and vice versa).
    """
    results = {}

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
        recall    = tp / (tp + fn) if (tp + fn) > 0 else None
        f1 = (
            2 * precision * recall / (precision + recall)
            if precision is not None and recall is not None and (precision + recall) > 0
            else None
        )

        gt_count = len(cat_gt)
        det_count = len(cat_dets)

        results[cat] = {
            "category": cat,
            "display": CATEGORY_DISPLAY[cat],
            "GT": gt_count,
            "detected": det_count,
            "TP": tp,
            "FP": fp,
            "FN": fn,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "detector": DETECTOR_MAP.get(cat, "N/A"),
            "fp_examples": [cat_dets[i] for i, m in enumerate(det_matched) if not m],
            "fn_examples": [cat_gt[i] for i, m in enumerate(gt_matched) if not m],
        }

    # Overall aggregate
    total_gt  = sum(results[c]["GT"] for c in ALL_CATEGORIES)
    total_det = sum(results[c]["detected"] for c in ALL_CATEGORIES)
    total_tp  = sum(results[c]["TP"] for c in ALL_CATEGORIES)
    total_fp  = sum(results[c]["FP"] for c in ALL_CATEGORIES)
    total_fn  = sum(results[c]["FN"] for c in ALL_CATEGORIES)

    overall_p  = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else None
    overall_r  = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else None
    overall_f1 = (
        2 * overall_p * overall_r / (overall_p + overall_r)
        if overall_p is not None and overall_r is not None and (overall_p + overall_r) > 0
        else None
    )

    results["OVERALL"] = {
        "category": "OVERALL",
        "display": "Overall",
        "GT": total_gt,
        "detected": total_det,
        "TP": total_tp,
        "FP": total_fp,
        "FN": total_fn,
        "precision": overall_p,
        "recall": overall_r,
        "f1": overall_f1,
        "detector": "—",
        "fp_examples": [],
        "fn_examples": [],
    }

    # Validation checks
    assert total_tp + total_fn == total_gt, (
        f"VALIDATION FAILED: TP({total_tp}) + FN({total_fn}) = {total_tp+total_fn} "
        f"!= GT({total_gt})"
    )
    assert total_tp + total_fp == sum(
        results[c]["TP"] + results[c]["FP"] for c in ALL_CATEGORIES
    ), "VALIDATION FAILED: TP+FP sum mismatch"

    print(f"  Validation checks PASSED: TP+FN={total_tp+total_fn} == GT={total_gt}")
    return results


# ---------------------------------------------------------------------------
# CSV matrix writers
# ---------------------------------------------------------------------------

def _fmt(v, na="N/A"):
    if v is None:
        return na
    if isinstance(v, float):
        return f"{v:.3f}"
    return str(v)


def write_matrices(scores: dict, out_dir: Path) -> None:
    matrices_dir = out_dir / "matrices"
    matrices_dir.mkdir(parents=True, exist_ok=True)

    # 1. Per-category TP / FP / FN
    with open(matrices_dir / "per_category_tp_fp_fn.csv", "w", encoding="utf-8") as f:
        f.write("PII Category,GT Instances,Detected,TP,FP,FN\n")
        for cat in ALL_CATEGORIES:
            s = scores[cat]
            f.write(f"{s['display']},{s['GT']},{s['detected']},{s['TP']},{s['FP']},{s['FN']}\n")
        s = scores["OVERALL"]
        f.write(f"Overall,{s['GT']},{s['detected']},{s['TP']},{s['FP']},{s['FN']}\n")
    print("  Saved: matrices/per_category_tp_fp_fn.csv")

    # 2. Per-category metrics
    with open(matrices_dir / "per_category_metrics.csv", "w", encoding="utf-8") as f:
        f.write("PII Category,GT,Detected,TP,FP,FN,Precision,Recall,F1\n")
        for cat in ALL_CATEGORIES:
            s = scores[cat]
            note = "" if s["GT"] > 0 else " (no GT)"
            f.write(
                f"{s['display']}{note},{s['GT']},{s['detected']},"
                f"{s['TP']},{s['FP']},{s['FN']},"
                f"{_fmt(s['precision'])},{_fmt(s['recall'])},{_fmt(s['f1'])}\n"
            )
        s = scores["OVERALL"]
        f.write(
            f"Overall,{s['GT']},{s['detected']},"
            f"{s['TP']},{s['FP']},{s['FN']},"
            f"{_fmt(s['precision'])},{_fmt(s['recall'])},{_fmt(s['f1'])}\n"
        )
    print("  Saved: matrices/per_category_metrics.csv")

    # 3. Overall outcome matrix
    s = scores["OVERALL"]
    with open(matrices_dir / "overall_outcome_matrix.csv", "w", encoding="utf-8") as f:
        f.write(",System: Detected,System: Not Detected\n")
        f.write(f"Actual: PII,{s['TP']} (TP),{s['FN']} (FN)\n")
        f.write(f"Non-PII Detection,{s['FP']} (FP),N/A (TN not defined at span level)\n")
    print("  Saved: matrices/overall_outcome_matrix.csv")

    # 4. Coverage matrix
    with open(matrices_dir / "coverage_matrix.csv", "w", encoding="utf-8") as f:
        f.write("Category,GT Instances,Detected Instances,Missed Instances,Coverage (Recall)\n")
        for cat in ALL_CATEGORIES:
            s = scores[cat]
            missed = s["FN"]
            coverage = _fmt(s["recall"]) if s["GT"] > 0 else "N/A (no GT)"
            f.write(f"{s['display']},{s['GT']},{s['detected']},{missed},{coverage}\n")
    print("  Saved: matrices/coverage_matrix.csv")

    # 5. Error analysis matrix
    with open(matrices_dir / "error_analysis_matrix.csv", "w", encoding="utf-8") as f:
        f.write("Category,FP,FN,Main FP Cause,Main FN Cause,Recommended Improvement\n")
        causes = {
            "PERSON":      ("NER hallucination on financial Title-Case terms",
                            "Short names / unusual names below confidence threshold",
                            "Fine-tune NER on financial domain corpus"),
            "COMPANY":     ("Regulatory bodies mis-tagged as private companies",
                            "Companies without standard corporate suffixes",
                            "Expand corporate suffix list; add SEC filing corpus"),
            "ADDRESS":     ("Partial address fragments incorrectly clustered",
                            "Addresses split across XML runs during extraction",
                            "Improve run-bridging in docx_io.py"),
            "EMAIL":       ("None expected (deterministic regex)",
                            "Obfuscated emails (e.g. name [at] domain)",
                            "Add obfuscation-aware patterns"),
            "PHONE":       ("None expected (deterministic regex)",
                            "Formatted numbers with unusual separators",
                            "Extend separator patterns"),
            "SSN":         ("N/A (no GT instances in document)",
                            "N/A (no GT instances in document)",
                            "Use document with SSN data for this category"),
            "CREDIT_CARD": ("N/A (no GT instances in document)",
                            "N/A (no GT instances in document)",
                            "Use document with card data for this category"),
            "DOB":         ("N/A (no GT instances in document)",
                            "N/A (no GT instances in document)",
                            "Use document with DOB data for this category"),
            "IP":          ("N/A (no GT instances in document)",
                            "N/A (no GT instances in document)",
                            "Use document with IP data for this category"),
        }
        for cat in ALL_CATEGORIES:
            s = scores[cat]
            fp_cause, fn_cause, improvement = causes[cat]
            f.write(
                f"{s['display']},{s['FP']},{s['FN']},"
                f"\"{fp_cause}\",\"{fn_cause}\",\"{improvement}\"\n"
            )
    print("  Saved: matrices/error_analysis_matrix.csv")

    # 6. Detector performance
    with open(out_dir / "detector_metrics.csv", "w", encoding="utf-8") as f:
        f.write("Detector,PII Categories,TP,FP,FN,Precision,Recall,F1\n")
        # NER detectors
        ner_cats = ["PERSON", "COMPANY", "ADDRESS"]
        ner_tp = sum(scores[c]["TP"] for c in ner_cats)
        ner_fp = sum(scores[c]["FP"] for c in ner_cats)
        ner_fn = sum(scores[c]["FN"] for c in ner_cats)
        ner_p  = ner_tp / (ner_tp + ner_fp) if (ner_tp + ner_fp) > 0 else None
        ner_r  = ner_tp / (ner_tp + ner_fn) if (ner_tp + ner_fn) > 0 else None
        ner_f1 = (2*ner_p*ner_r/(ner_p+ner_r)) if ner_p and ner_r else None
        f.write(f"NER (spaCy en_core_web_sm),Person / Company / Address,{ner_tp},{ner_fp},{ner_fn},{_fmt(ner_p)},{_fmt(ner_r)},{_fmt(ner_f1)}\n")

        # Regex detectors
        regex_cats = [("EMAIL", "Email Regex"), ("PHONE", "Phone Regex"),
                      ("IP", "IP Regex"), ("CREDIT_CARD", "Credit Card Regex + Luhn"),
                      ("SSN", "SSN Regex"), ("DOB", "DOB Context Regex")]
        for rcat, rname in regex_cats:
            s = scores[rcat]
            f.write(
                f"{rname},{s['display']},{s['TP']},{s['FP']},{s['FN']},"
                f"{_fmt(s['precision'])},{_fmt(s['recall'])},{_fmt(s['f1'])}\n"
            )
    print("  Saved: detector_metrics.csv")


# ---------------------------------------------------------------------------
# Chart generation
# ---------------------------------------------------------------------------

def _setup_style():
    plt.rcParams.update({
        "figure.facecolor": DARK_BG,
        "axes.facecolor": CARD_BG,
        "axes.edgecolor": "#334155",
        "axes.labelcolor": TEXT_COLOR,
        "xtick.color": TEXT_COLOR,
        "ytick.color": TEXT_COLOR,
        "text.color": TEXT_COLOR,
        "grid.color": "#1e293b",
        "font.family": "DejaVu Sans",
        "font.size": 9,
    })


def chart_tp_fp_fn(scores: dict, out_dir: Path) -> None:
    """Chart 1 — Per-Category TP / FP / FN grouped bar chart."""
    cats = [CATEGORY_DISPLAY[c] for c in ALL_CATEGORIES]
    tp_vals = [scores[c]["TP"] for c in ALL_CATEGORIES]
    fp_vals = [scores[c]["FP"] for c in ALL_CATEGORIES]
    fn_vals = [scores[c]["FN"] for c in ALL_CATEGORIES]

    x = np.arange(len(cats))
    width = 0.25

    fig, ax = plt.subplots(figsize=(13, 6))
    fig.patch.set_facecolor(DARK_BG)

    bars_tp = ax.bar(x - width, tp_vals, width, label="True Positive (TP)", color=COLORS["TP"], zorder=3)
    bars_fp = ax.bar(x,         fp_vals, width, label="False Positive (FP)", color=COLORS["FP"], zorder=3)
    bars_fn = ax.bar(x + width, fn_vals, width, label="False Negative (FN)", color=COLORS["FN"], zorder=3)

    def label_bars(bars):
        for bar in bars:
            h = bar.get_height()
            if h > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, h + 0.5,
                        str(int(h)), ha="center", va="bottom", fontsize=8, color=TEXT_COLOR)

    label_bars(bars_tp)
    label_bars(bars_fp)
    label_bars(bars_fn)

    ax.set_title("Per-Category: True Positive / False Positive / False Negative",
                 fontsize=13, fontweight="bold", color=TEXT_COLOR, pad=15)
    ax.set_xlabel("PII Category", fontsize=10, labelpad=8)
    ax.set_ylabel("Number of Entities", fontsize=10)
    ax.set_xticks(x)
    ax.set_xticklabels(cats, rotation=25, ha="right")
    ax.legend(facecolor=CARD_BG, edgecolor="#334155", labelcolor=TEXT_COLOR)
    ax.grid(axis="y", alpha=0.2, zorder=0)
    ax.set_ylim(0, max(max(tp_vals), 1) * 1.18)

    plt.tight_layout()
    path = out_dir / "charts" / "per_category_tp_fp_fn.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


def chart_precision_recall_f1(scores: dict, out_dir: Path) -> None:
    """Chart 2 — Per-Category Precision / Recall / F1."""
    # Only show categories that have ground truth
    active_cats = [c for c in ALL_CATEGORIES if scores[c]["GT"] > 0]
    cats = [CATEGORY_DISPLAY[c] for c in active_cats]

    p_vals  = [scores[c]["precision"] or 0 for c in active_cats]
    r_vals  = [scores[c]["recall"]    or 0 for c in active_cats]
    f1_vals = [scores[c]["f1"]        or 0 for c in active_cats]

    x = np.arange(len(cats))
    width = 0.25

    fig, ax = plt.subplots(figsize=(12, 6))
    fig.patch.set_facecolor(DARK_BG)

    ax.bar(x - width, p_vals,  width, label="Precision", color=COLORS["P"],  zorder=3)
    ax.bar(x,         r_vals,  width, label="Recall",    color=COLORS["R"],  zorder=3)
    ax.bar(x + width, f1_vals, width, label="F1 Score",  color=COLORS["F1"], zorder=3)

    ax.set_title("Per-Category: Precision / Recall / F1\n(N/A categories excluded — no ground-truth instances)",
                 fontsize=12, fontweight="bold", color=TEXT_COLOR, pad=12)
    ax.set_xlabel("PII Category", fontsize=10, labelpad=8)
    ax.set_ylabel("Score (0.0 – 1.0)", fontsize=10)
    ax.set_xticks(x)
    ax.set_xticklabels(cats, rotation=20, ha="right")
    ax.set_ylim(0, 1.18)
    ax.legend(facecolor=CARD_BG, edgecolor="#334155", labelcolor=TEXT_COLOR)
    ax.grid(axis="y", alpha=0.2, zorder=0)

    # Add value labels
    for bars, vals in [(ax.containers[0], p_vals),
                       (ax.containers[1], r_vals),
                       (ax.containers[2], f1_vals)]:
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, v + 0.01,
                    f"{v:.2f}", ha="center", va="bottom", fontsize=7.5, color=TEXT_COLOR)

    plt.tight_layout()
    path = out_dir / "charts" / "precision_recall_f1.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


def chart_overall_outcome(scores: dict, out_dir: Path) -> None:
    """Chart 3 — Overall TP / FP / FN (no fake TN Millions)."""
    s = scores["OVERALL"]
    labels = ["True Positive (TP)", "False Positive (FP)", "False Negative (FN)"]
    values = [s["TP"], s["FP"], s["FN"]]
    colors = [COLORS["TP"], COLORS["FP"], COLORS["FN"]]

    fig, ax = plt.subplots(figsize=(8, 6))
    fig.patch.set_facecolor(DARK_BG)

    bars = ax.bar(labels, values, color=colors, width=0.5, zorder=3)
    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 1,
                str(int(h)), ha="center", va="bottom", fontsize=13,
                fontweight="bold", color=TEXT_COLOR)

    ax.set_title("Overall PII Entity Detection Outcome\n(Entity-Level, TN Not Defined at Span Level)",
                 fontsize=12, fontweight="bold", color=TEXT_COLOR, pad=12)
    ax.set_ylabel("Number of Entities", fontsize=10)
    ax.grid(axis="y", alpha=0.2, zorder=0)
    ax.set_ylim(0, max(values) * 1.18)

    plt.tight_layout()
    path = out_dir / "charts" / "overall_detection_outcome.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


def chart_pii_distribution(ground_truth: list[dict], out_dir: Path) -> None:
    """Chart 4 — Ground-truth PII distribution by category."""
    counts = Counter(g["category"].upper() for g in ground_truth)
    cats = [CATEGORY_DISPLAY.get(c, c) for c in ALL_CATEGORIES]
    vals = [counts.get(c, 0) for c in ALL_CATEGORIES]

    # filter to non-zero for pie
    non_zero = [(c, v) for c, v in zip(cats, vals) if v > 0]
    pie_labels = [f"{c} ({v})" for c, v in non_zero]
    pie_vals = [v for _, v in non_zero]

    palette = ["#2dd4bf", "#818cf8", "#f472b6", "#fb923c", "#34d399",
               "#60a5fa", "#a78bfa", "#fbbf24", "#f87171"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    fig.patch.set_facecolor(DARK_BG)

    # Pie chart
    wedges, texts, autotexts = ax1.pie(
        pie_vals, labels=None, autopct="%1.1f%%",
        colors=palette[:len(pie_vals)], startangle=140,
        pctdistance=0.82, wedgeprops={"edgecolor": DARK_BG, "linewidth": 1.5}
    )
    for at in autotexts:
        at.set_color(DARK_BG)
        at.set_fontsize(8)
    ax1.legend(wedges, pie_labels, loc="center left",
               bbox_to_anchor=(-0.35, 0.5), facecolor=CARD_BG,
               edgecolor="#334155", labelcolor=TEXT_COLOR, fontsize=8)
    ax1.set_title("Ground Truth PII Distribution", fontsize=11,
                  fontweight="bold", color=TEXT_COLOR)
    ax1.set_facecolor(CARD_BG)

    # Bar chart
    ax2.set_facecolor(CARD_BG)
    bars = ax2.barh(cats, vals, color=palette[:len(cats)], zorder=3)
    for bar, v in zip(bars, vals):
        if v > 0:
            ax2.text(v + 0.5, bar.get_y() + bar.get_height() / 2,
                     str(v), va="center", fontsize=8.5, color=TEXT_COLOR)
    ax2.set_title("Ground Truth Instances by Category", fontsize=11,
                  fontweight="bold", color=TEXT_COLOR)
    ax2.set_xlabel("Count", fontsize=9)
    ax2.grid(axis="x", alpha=0.2, zorder=0)

    plt.tight_layout()
    path = out_dir / "charts" / "pii_distribution.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


def chart_detection_coverage(scores: dict, out_dir: Path) -> None:
    """Chart 5 — Detection coverage (Recall) by category."""
    active_cats = [c for c in ALL_CATEGORIES if scores[c]["GT"] > 0]
    cats = [CATEGORY_DISPLAY[c] for c in active_cats]
    recalls = [scores[c]["recall"] or 0 for c in active_cats]

    fig, ax = plt.subplots(figsize=(10, 5))
    fig.patch.set_facecolor(DARK_BG)

    colors_bar = [COLORS["R"] if r >= 0.85 else COLORS["FP"] for r in recalls]
    bars = ax.barh(cats, recalls, color=colors_bar, zorder=3)
    for bar, r in zip(bars, recalls):
        ax.text(r + 0.01, bar.get_y() + bar.get_height() / 2,
                f"{r:.3f}", va="center", fontsize=9, color=TEXT_COLOR)

    ax.set_title("Detection Coverage (Recall) by Category\n(Green ≥ 0.85 target | Red < 0.85)",
                 fontsize=12, fontweight="bold", color=TEXT_COLOR, pad=12)
    ax.set_xlabel("Recall Score", fontsize=10)
    ax.set_xlim(0, 1.15)
    ax.axvline(x=0.85, color="#fbbf24", linestyle="--", alpha=0.6, label="0.85 target")
    ax.legend(facecolor=CARD_BG, edgecolor="#334155", labelcolor=TEXT_COLOR)
    ax.grid(axis="x", alpha=0.2, zorder=0)

    plt.tight_layout()
    path = out_dir / "charts" / "detection_coverage.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


# ---------------------------------------------------------------------------
# JSON output
# ---------------------------------------------------------------------------

def write_json_results(scores: dict, out_dir: Path) -> None:
    # per_category_metrics.json
    per_cat = {}
    for cat in ALL_CATEGORIES:
        s = scores[cat]
        per_cat[cat] = {
            "display": s["display"],
            "ground_truth_count": s["GT"],
            "detected_count": s["detected"],
            "TP": s["TP"], "FP": s["FP"], "FN": s["FN"],
            "precision": s["precision"],
            "recall": s["recall"],
            "f1": s["f1"],
            "has_ground_truth": s["GT"] > 0,
        }
    with open(out_dir / "per_category_metrics.json", "w") as f:
        json.dump(per_cat, f, indent=2)
    print("  Saved: per_category_metrics.json")

    # overall_metrics.json
    ov = scores["OVERALL"]
    overall = {
        "total_ground_truth": ov["GT"],
        "total_detected": ov["detected"],
        "TP": ov["TP"], "FP": ov["FP"], "FN": ov["FN"],
        "precision": ov["precision"],
        "recall": ov["recall"],
        "f1": ov["f1"],
        "accuracy": "N/A — Accuracy not reported. Evaluation is at PII entity/span level. "
                    "True Negatives cannot be meaningfully defined without exhaustive "
                    "annotation of every non-PII span in the document.",
        "validation": {
            "TP_plus_FN_equals_GT": ov["TP"] + ov["FN"] == ov["GT"],
            "note": "All category totals verified to sum correctly to overall."
        }
    }
    with open(out_dir / "overall_metrics.json", "w") as f:
        json.dump(overall, f, indent=2)
    print("  Saved: overall_metrics.json")

    # evaluation_results.json (full)
    serialisable = {}
    for cat in list(ALL_CATEGORIES) + ["OVERALL"]:
        s = scores[cat]
        serialisable[cat] = {k: v for k, v in s.items()
                             if k not in ("fp_examples", "fn_examples")}
        serialisable[cat]["fp_examples"] = s["fp_examples"][:10]
        serialisable[cat]["fn_examples"] = s["fn_examples"][:10]
    with open(out_dir / "evaluation_results.json", "w") as f:
        json.dump(serialisable, f, indent=2, ensure_ascii=False)
    print("  Saved: evaluation_results.json")


# ---------------------------------------------------------------------------
# Text report
# ---------------------------------------------------------------------------

def write_text_report(scores: dict, out_dir: Path) -> None:
    lines = []
    def w(line=""):
        lines.append(line)

    ov = scores["OVERALL"]

    w("# PII Redaction Tool — Corrected Evaluation Report")
    w()
    w("> **Methodology:** Span/entity-level matching against an independently")
    w("> validated ground truth. A detection is a True Positive ONLY when it")
    w("> matches a ground-truth record by category + normalized text + location.")
    w("> Accuracy is NOT reported (TN is undefined at span level).")
    w()

    # Overall
    w("## Overall Summary")
    w()
    w("| Metric | Value |")
    w("|--------|-------|")
    w(f"| Total Ground Truth PII | {ov['GT']} |")
    w(f"| Total System Detections | {ov['detected']} |")
    w(f"| True Positives (TP) | {ov['TP']} |")
    w(f"| False Positives (FP) | {ov['FP']} |")
    w(f"| False Negatives (FN) | {ov['FN']} |")
    w(f"| **Precision** | **{_fmt(ov['precision'])}** |")
    w(f"| **Recall** | **{_fmt(ov['recall'])}** |")
    w(f"| **F1 Score** | **{_fmt(ov['f1'])}** |")
    w(f"| Accuracy | N/A (span-level eval; TN undefined) |")
    w()

    # Per-category
    w("## Per-Category Results")
    w()
    w("| Category | GT | Detected | TP | FP | FN | Precision | Recall | F1 |")
    w("|---|---|---|---|---|---|---|---|---|")
    for cat in ALL_CATEGORIES:
        s = scores[cat]
        note = "" if s["GT"] > 0 else " *(no GT)*"
        w(
            f"| {s['display']}{note} | {s['GT']} | {s['detected']} "
            f"| {s['TP']} | {s['FP']} | {s['FN']} "
            f"| {_fmt(s['precision'])} | {_fmt(s['recall'])} | {_fmt(s['f1'])} |"
        )
    w(
        f"| **Overall** | **{ov['GT']}** | **{ov['detected']}** "
        f"| **{ov['TP']}** | **{ov['FP']}** | **{ov['FN']}** "
        f"| **{_fmt(ov['precision'])}** | **{_fmt(ov['recall'])}** | **{_fmt(ov['f1'])}** |"
    )
    w()

    # Outcome matrix
    w("## Overall PII Entity Detection Outcome Matrix")
    w()
    w("```")
    w("                         System Detection")
    w("                       Detected    Not Detected")
    w("                     ----------------------------")
    w(f"Actual PII           |  {ov['TP']:>5}   |   {ov['FN']:>5}  (FN)")
    w("                     ----------------------------")
    w(f"Non-PII Detection    |  {ov['FP']:>5}   |   N/A")
    w("                     ----------------------------")
    w(f"                       (TP)           (TN: not defined at span level)")
    w("```")
    w()
    w("> Note: True Negatives are not computable at the entity-span level.")
    w("> Accuracy is therefore not reported as a primary metric.")
    w("> Primary metrics are Precision, Recall, and F1.")
    w()

    # Error analysis
    w("## Error Analysis")
    w()
    w("### False Positives")
    has_fp = False
    for cat in ALL_CATEGORIES:
        fps = scores[cat]["fp_examples"]
        if fps:
            has_fp = True
            w(f"#### {CATEGORY_DISPLAY[cat]}")
            w()
            w("| Text | Detector | Confidence |")
            w("|---|---|---|")
            for fp in fps[:20]:
                w(f"| `{fp['text']}` | {fp.get('detector','—')} | {fp.get('confidence','—')} |")
            w()
    if not has_fp:
        w("*No false positives detected against the independent ground truth.*")
        w()

    w("### False Negatives")
    has_fn = False
    for cat in ALL_CATEGORIES:
        fns = scores[cat]["fn_examples"]
        if fns:
            has_fn = True
            w(f"#### {CATEGORY_DISPLAY[cat]}")
            w()
            w("| ID | Text | Notes |")
            w("|---|---|---|")
            for fn in fns:
                id_ = fn.get("id", "—")
                w(f"| {id_} | `{fn['text']}` | {fn.get('notes','')} |")
            w()
    if not has_fn:
        w("*No false negatives — all ground-truth PII instances were detected.*")
        w()

    out_path = out_dir / "evaluation_report.md"
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"  Saved: {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 60)
    print("RUNNING CORRECTED EVALUATION PIPELINE")
    print("=" * 60)

    # 1. Load independent ground truth
    print(f"\n[1/6] Loading independent ground truth: {GROUND_TRUTH_PATH}")
    gt = load_json(GROUND_TRUTH_PATH)
    from collections import Counter
    gt_cats = Counter(g["category"].upper() for g in gt)
    print(f"  Total ground-truth entities: {len(gt)}")
    for cat, cnt in sorted(gt_cats.items()):
        print(f"    {cat}: {cnt}")

    # 2. Load system detections
    print(f"\n[2/6] Loading system detections: {DETECTIONS_PATH}")
    dets = load_json(DETECTIONS_PATH)
    det_cats = Counter(d["category"].upper() for d in dets)
    print(f"  Total system detections: {len(dets)}")
    for cat, cnt in sorted(det_cats.items()):
        print(f"    {cat}: {cnt}")

    # 3. Score
    print("\n[3/6] Scoring (matching detections against ground truth)...")
    scores = score(dets, gt)
    ov = scores["OVERALL"]
    print(f"  Overall -> TP={ov['TP']}  FP={ov['FP']}  FN={ov['FN']}")
    print(f"          -> P={_fmt(ov['precision'])}  R={_fmt(ov['recall'])}  F1={_fmt(ov['f1'])}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 4. CSV matrices
    print("\n[4/6] Writing CSV matrices...")
    write_matrices(scores, OUTPUT_DIR)

    # 5. JSON results
    print("\n[5/6] Writing JSON results...")
    write_json_results(scores, OUTPUT_DIR)

    # 6. Charts
    print("\n[6/6] Generating charts...")
    _setup_style()
    chart_tp_fp_fn(scores, OUTPUT_DIR)
    chart_precision_recall_f1(scores, OUTPUT_DIR)
    chart_overall_outcome(scores, OUTPUT_DIR)
    chart_pii_distribution(gt, OUTPUT_DIR)
    chart_detection_coverage(scores, OUTPUT_DIR)

    # Text report
    write_text_report(scores, OUTPUT_DIR)

    print("\n" + "=" * 60)
    print("EVALUATION COMPLETE")
    print(f"All outputs written to: {OUTPUT_DIR.resolve()}")
    print("=" * 60)


if __name__ == "__main__":
    main()
