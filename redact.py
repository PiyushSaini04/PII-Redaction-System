"""
redact.py
---------
CLI entry point for the PII Redaction Tool.

Usage
-----
# Full redaction
python redact.py --input "data/input/Red Herring Prospectus.docx" \
                 --output "data/output/redacted_prospectus.docx" \
                 --seed 42

# Dry-run: emit detections.json only, skip writing redacted DOCX
python redact.py --input "data/input/Red Herring Prospectus.docx" \
                 --dry-run

# Run only a subset of categories
python redact.py --input "data/input/Red Herring Prospectus.docx" \
                 --categories "PERSON,EMAIL,PHONE"

Pipeline
--------
load_document
  → extract_all_text_blocks
    → collect_all_matches (regex + NER, resolve_overlaps)
      → build_mapping (seeded Faker)
        → [dry-run: emit detections.json]
        → [normal: apply_to_docx, save]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from docx_io import extract_all_text_blocks, load_document, save_document
from replacer import apply_to_docx, build_mapping
from resolver import collect_all_matches

# All 9 required PII categories
ALL_CATEGORIES = {
    "PERSON", "EMAIL", "PHONE", "COMPANY",
    "ADDRESS", "SSN", "CREDIT_CARD", "DOB", "IP",
}


# ---------------------------------------------------------------------------
# detections.json serialisation
# ---------------------------------------------------------------------------

def _match_to_dict(match) -> dict:
    """Serialise a PIIMatch to a JSON-safe dict (audit/debug artifact)."""
    loc = match.source_block.location
    # Strip non-serialisable Run objects; keep source metadata only
    safe_loc = {k: v for k, v in loc.items() if k != "runs"}
    return {
        "text": match.text,
        "category": match.category,
        "start_char": match.start_char,
        "end_char": match.end_char,
        "source_location": safe_loc,
        "confidence": round(match.confidence, 4),
        "needs_review": match.needs_review,
        "detector": match.detector,
    }


def write_detections_json(matches, output_path: Path) -> None:
    """Write all detection occurrences to detections.json."""
    data = [_match_to_dict(m) for m in matches]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"  detections.json -> {output_path}  ({len(data)} occurrences)")


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run_pipeline(
    input_path: Path,
    output_path: Path,
    seed: int,
    dry_run: bool,
    categories: set[str],
    detections_path: Path,
) -> None:
    print(f"\n[1/5] Loading document: {input_path}")
    doc = load_document(input_path)

    print("[2/5] Extracting text blocks …")
    blocks = extract_all_text_blocks(doc)
    print(f"      {len(blocks)} text blocks extracted")

    print("[3/5] Running detectors …")
    matches = collect_all_matches(blocks, categories=categories)
    # Per-category summary
    from collections import Counter
    cat_counts = Counter(m.category for m in matches)
    for cat in sorted(cat_counts):
        print(f"      {cat}: {cat_counts[cat]} occurrence(s)")
    print(f"      Total: {len(matches)} occurrences across all categories")

    print("[4/5] Building replacement mapping …")
    mapping = build_mapping(matches, seed=seed)
    print(f"      {len(mapping)} unique PII value(s) -> fake value(s)")

    # Always write detections.json so it can be used for evaluation
    write_detections_json(matches, detections_path)

    if dry_run:
        print("\n[dry-run] Skipping DOCX rewriting. detections.json written above.")
        return

    print("[5/5] Applying replacements to DOCX and saving …")
    apply_to_docx(matches, mapping)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_document(doc, output_path)
    print(f"      Saved: {output_path}")

    print("\n Done.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="redact",
        description="PII Detection & Redaction Tool — detects and redacts "
                    "PII in a .docx file using hybrid regex + spaCy NER.",
    )
    parser.add_argument(
        "--input", "-i",
        required=True,
        help="Path to the source .docx file.",
    )
    parser.add_argument(
        "--output", "-o",
        default="data/output/redacted_prospectus.docx",
        help="Path for the redacted .docx output. "
             "(default: data/output/redacted_prospectus.docx)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for deterministic fake-value generation. (default: 42)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Emit detections.json only; do not write the redacted DOCX.",
    )
    parser.add_argument(
        "--categories",
        default=None,
        help="Comma-separated subset of PII categories to run. "
             f"(default: all — {', '.join(sorted(ALL_CATEGORIES))})",
    )
    parser.add_argument(
        "--detections-out",
        default="data/output/detections.json",
        help="Path for the detections.json audit artifact. "
             "(default: data/output/detections.json)",
    )
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"ERROR: Input file not found: {input_path}", file=sys.stderr)
        return 1

    # Resolve categories
    if args.categories:
        requested = {c.strip().upper() for c in args.categories.split(",")}
        unknown = requested - ALL_CATEGORIES
        if unknown:
            print(
                f"ERROR: Unknown categories: {unknown}. "
                f"Valid: {ALL_CATEGORIES}",
                file=sys.stderr,
            )
            return 1
        categories = requested
    else:
        categories = ALL_CATEGORIES

    try:
        run_pipeline(
            input_path=input_path,
            output_path=Path(args.output),
            seed=args.seed,
            dry_run=args.dry_run,
            categories=categories,
            detections_path=Path(args.detections_out),
        )
    except Exception as exc:
        print(f"\nERROR: {exc}", file=sys.stderr)
        raise

    return 0


if __name__ == "__main__":
    sys.exit(main())
