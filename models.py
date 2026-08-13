"""
models.py
---------
Shared data models for the PII Redaction Tool.

TextBlock  — a contiguous span of text extracted from one DOCX source location,
             carrying a reference to the original python-docx Run objects so that
             character-offset-based replacement can locate and modify the correct runs.

PIIMatch   — a single OCCURRENCE of a detected PII entity.  Multiple PIIMatch
             objects may share the same (text, category) pair — one per occurrence
             in the document.  They are never collapsed during detection; the
             replacement layer reads every occurrence independently.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class TextBlock:
    """
    One contiguous block of text from a single DOCX source location.

    Attributes
    ----------
    text : str
        The combined text of all runs in this block
        (equivalent to paragraph.text / cell.text).
    location : dict
        Source information.  Keys depend on block type:

        Paragraph
        ---------
        {
            "type": "paragraph",
            "para_idx": int,          # index in doc.paragraphs
            "runs": list[Run],        # python-docx Run objects
            "run_offsets": list[int], # cumulative start-char of each run in text
        }

        Table cell
        ----------
        {
            "type":      "table_cell",
            "table_idx": int,
            "row_idx":   int,
            "col_idx":   int,
            "para_idx":  int,         # paragraph within the cell (0-based)
            "runs":      list[Run],
            "run_offsets": list[int],
        }

        Header / Footer
        ---------------
        {
            "type":        "header" | "footer",
            "section_idx": int,
            "para_idx":    int,
            "runs":        list[Run],
            "run_offsets": list[int],
        }

    Notes
    -----
    run_offsets[i] is the character index in `text` where runs[i] starts.
    len(run_offsets) == len(runs).
    """

    text: str
    location: dict[str, Any]


@dataclass
class PIIMatch:
    """
    A single OCCURRENCE of a detected PII entity in the document.

    If the same PII value appears N times, the detector produces N PIIMatch
    objects — one per occurrence.  The replacement system (replacer.py) uses
    build_mapping() to assign one canonical fake value per unique original
    text, and then looks that value up for every occurrence.

    Attributes
    ----------
    text : str
        Exact matched substring (preserves original casing/punctuation).
    category : str
        One of the 9 required labels:
        PERSON | EMAIL | PHONE | COMPANY | ADDRESS |
        SSN | CREDIT_CARD | DOB | IP
    start_char : int
        Start character offset within source_block.text.
    end_char : int
        End character offset (exclusive) within source_block.text.
    source_block : TextBlock
        Reference to the TextBlock from which this match was extracted.
    confidence : float
        Detection confidence in [0, 1].
        1.0 for regex/high-confidence NER matches;
        < 1.0 for heuristic matches (ADDRESS, DOB).
    needs_review : bool
        True for low-confidence heuristic matches (primarily ADDRESS).
    detector : str
        "regex" | "ner" | "heuristic"
    """

    text: str
    category: str
    start_char: int
    end_char: int
    source_block: TextBlock
    confidence: float = 1.0
    needs_review: bool = False
    detector: str = ""
