"""
resolver.py
-----------
Merge regex + NER detector outputs into a single, conflict-free list of
PIIMatch occurrences.

All occurrence records are PRESERVED — this module never collapses multiple
occurrences of the same PII value.  Deduplication into a canonical replacement
mapping happens later in replacer.py::build_mapping().

Public API
----------
resolve_overlaps(matches)        -> list[PIIMatch]
collect_all_matches(blocks)      -> list[PIIMatch]
"""

from __future__ import annotations

from detectors.ner_detectors import run_all_ner_detectors
from detectors.regex_detectors import run_all_regex_detectors
from models import PIIMatch, TextBlock


# ---------------------------------------------------------------------------
# Overlap resolution
# ---------------------------------------------------------------------------

def _spans_overlap(a: PIIMatch, b: PIIMatch) -> bool:
    """Return True when two matches from the same block overlap."""
    if a.source_block is not b.source_block:
        return False
    return a.start_char < b.end_char and b.start_char < a.end_char


def _prefer(a: PIIMatch, b: PIIMatch) -> PIIMatch:
    """
    When two matches overlap, return the preferred one.

    Rules (in order):
      1. Regex beats NER/heuristic  — regex is higher-precision for
         structured PII; NER sometimes captures a superset span containing
         a credit-card or SSN.
      2. Longer span wins — when both are the same detector type, keep the
         more complete match.
      3. Higher confidence wins as a tiebreaker.
    """
    detector_rank = {"regex": 0, "ner": 1, "heuristic": 2}
    rank_a = detector_rank.get(a.detector, 1)
    rank_b = detector_rank.get(b.detector, 1)

    if rank_a != rank_b:
        return a if rank_a < rank_b else b

    len_a = a.end_char - a.start_char
    len_b = b.end_char - b.start_char
    if len_a != len_b:
        return a if len_a > len_b else b

    return a if a.confidence >= b.confidence else b


def resolve_overlaps(matches: list[PIIMatch]) -> list[PIIMatch]:
    """
    Remove conflicting overlapping matches using the preference rules above.

    Each (block, span) appears at most once in the output.  Non-overlapping
    matches are always kept, including all N occurrences of the same text
    in different blocks (or non-overlapping spans within the same block).

    Algorithm: greedy scan over matches sorted by (block_id, start_char).
    """
    if not matches:
        return []

    # Sort by identity of source block (by id), then by start offset
    sorted_matches = sorted(
        matches,
        key=lambda m: (id(m.source_block), m.start_char, m.end_char),
    )

    resolved: list[PIIMatch] = [sorted_matches[0]]

    for candidate in sorted_matches[1:]:
        last = resolved[-1]
        if _spans_overlap(candidate, last):
            # Replace last with winner of preference comparison
            resolved[-1] = _prefer(last, candidate)
        else:
            resolved.append(candidate)

    return resolved


# ---------------------------------------------------------------------------
# Full detection pass over all blocks
# ---------------------------------------------------------------------------

def collect_all_matches(
    blocks: list[TextBlock],
    categories: set[str] | None = None,
) -> list[PIIMatch]:
    """
    Run all detectors (regex first, then NER) on every TextBlock and return
    a resolved list of PIIMatch occurrences.

    Parameters
    ----------
    blocks     : All TextBlocks from extract_all_text_blocks().
    categories : Optional set of category labels to run.  When None, all
                 9 required categories are run.  Supports --categories CLI flag.

    Returns
    -------
    Resolved list of PIIMatch objects — one per occurrence, no overlaps,
    all occurrences of repeated values preserved.
    """
    all_matches: list[PIIMatch] = []

    for block in blocks:
        block_matches: list[PIIMatch] = []

        # --- Regex detectors (run first for overlap precedence) ---
        regex_matches = run_all_regex_detectors(block)
        block_matches.extend(regex_matches)

        # --- NER detectors ---
        ner_matches = run_all_ner_detectors(block)
        block_matches.extend(ner_matches)

        # Resolve overlaps within this block
        resolved = resolve_overlaps(block_matches)

        # Filter by requested categories
        if categories:
            resolved = [m for m in resolved if m.category in categories]

        all_matches.extend(resolved)

    return all_matches
