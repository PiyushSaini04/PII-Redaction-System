"""
detectors/regex_detectors.py
-----------------------------
Regex-based detectors for structured, pattern-predictable PII.

Each public function accepts a TextBlock and returns list[PIIMatch].
Regex matches take precedence over NER matches on overlapping spans
(enforced in resolver.py).

Detectors
---------
detect_emails(block)          -> list[PIIMatch]   category="EMAIL"
detect_phone_numbers(block)   -> list[PIIMatch]   category="PHONE"
detect_ip_addresses(block)    -> list[PIIMatch]   category="IP"
detect_credit_cards(block)    -> list[PIIMatch]   category="CREDIT_CARD"
detect_ssns(block)            -> list[PIIMatch]   category="SSN"
detect_dates(block)           -> list[dict]       INTERNAL HELPER ONLY
detect_dates_of_birth(block)  -> list[PIIMatch]   category="DOB"

Note on PAN / Aadhaar
---------------------
Indian PAN numbers (AAAAA9999A) and Aadhaar numbers (\\d{4} \\d{4} \\d{4})
are explicitly outside the 9 required categories of this assignment.
They are NOT detected or redacted.
"""

from __future__ import annotations

import re
from typing import Any

from models import PIIMatch, TextBlock

# ---------------------------------------------------------------------------
# Email
# ---------------------------------------------------------------------------

_EMAIL_RE = re.compile(
    r"""
    (?<![/@\w])             # not preceded by @, /, or word char
    [\w.+\-]+               # local part
    @
    [\w\-]+                 # domain name
    (?:\.[\w\-]+)+          # one or more TLD components
    (?![.\w])               # not followed by word char or dot
    """,
    re.VERBOSE,
)


def detect_emails(block: TextBlock) -> list[PIIMatch]:
    """
    Detect email addresses using RFC-5322-inspired regex.

    Positive examples : cs.connect@kshinternational.com
    Negative examples : missing@, @nodomain, not_an_email
    """
    matches: list[PIIMatch] = []
    for m in _EMAIL_RE.finditer(block.text):
        matches.append(PIIMatch(
            text=m.group(),
            category="EMAIL",
            start_char=m.start(),
            end_char=m.end(),
            source_block=block,
            confidence=1.0,
            detector="regex",
        ))
    return matches


# ---------------------------------------------------------------------------
# Phone numbers
# ---------------------------------------------------------------------------

# Pattern priorities (tried in order, first match wins for a given span):
#   1. International with country code:  +CC <digits with spaces/hyphens>
#   2. Indian 10-digit bare (starts 6-9): 6–9 followed by 9 digits
#   3. US-style:  (NXX) NXX-XXXX  or  NXX-NXX-XXXX

_PHONE_INTL_RE = re.compile(
    r"""
    (?<!\d)
    \+\d{1,3}               # country code
    [\s\-\.]?
    \(?\d{1,5}\)?           # area/city code
    [\s\-\.]
    \d{3,5}                 # subscriber part 1
    [\s\-\.]
    \d{4,5}                 # subscriber part 2
    (?!\d)
    """,
    re.VERBOSE,
)

_PHONE_INDIA_RE = re.compile(
    r"""
    (?<!\d)
    [6-9]\d{9}              # Indian mobile (bare, 10 digits starting 6-9)
    (?!\d)
    """,
    re.VERBOSE,
)

_PHONE_US_RE = re.compile(
    r"""
    (?<!\d)
    (?:\(\d{3}\)|\d{3})     # area code
    [\s\-\.]
    \d{3}                   # exchange
    [\s\-\.]
    \d{4}                   # subscriber
    (?!\d)
    """,
    re.VERBOSE,
)

# Minimum 7 digits, max 15 (E.164)
_MIN_DIGITS = 7
_MAX_DIGITS = 15


def _digit_count(s: str) -> int:
    return sum(1 for c in s if c.isdigit())


def detect_phone_numbers(block: TextBlock) -> list[PIIMatch]:
    """
    Detect phone numbers in common international, Indian, and US formats.

    Applies length constraints (7–15 digits after stripping formatting)
    to avoid matching arbitrary financial reference numbers.

    Positive: +91 20 4505 3237, (555) 123-4567, 9876543210
    Negative: 123 (too short), 9999999999999999 (too long)
    """
    text = block.text
    seen_spans: set[tuple[int, int]] = set()
    matches: list[PIIMatch] = []

    for pattern in (_PHONE_INTL_RE, _PHONE_INDIA_RE, _PHONE_US_RE):
        for m in pattern.finditer(text):
            span = (m.start(), m.end())
            if span in seen_spans:
                continue
            # Overlap check against already-found spans
            if any(s <= m.start() < e or s < m.end() <= e
                   for s, e in seen_spans):
                continue
            digits = _digit_count(m.group())
            if _MIN_DIGITS <= digits <= _MAX_DIGITS:
                seen_spans.add(span)
                matches.append(PIIMatch(
                    text=m.group(),
                    category="PHONE",
                    start_char=m.start(),
                    end_char=m.end(),
                    source_block=block,
                    confidence=1.0,
                    detector="regex",
                ))

    return matches


# ---------------------------------------------------------------------------
# IP addresses
# ---------------------------------------------------------------------------

_IP_CANDIDATE_RE = re.compile(
    r"""
    (?<!\d\.)
    (?<!\d)
    (\d{1,3})\.(\d{1,3})\.(\d{1,3})\.(\d{1,3})
    (?!\.\d)
    (?!\d)
    """,
    re.VERBOSE,
)


def detect_ip_addresses(block: TextBlock) -> list[PIIMatch]:
    """
    Detect IPv4 addresses.  After regex match, validate that every octet
    is in [0, 255].  Rejects values like 999.999.999.999.

    Positive: 192.168.1.1, 10.0.0.1
    Negative: 999.1.1.1, 256.0.0.0, 1.2.3 (not 4 octets)
    """
    matches: list[PIIMatch] = []
    for m in _IP_CANDIDATE_RE.finditer(block.text):
        octets = [int(m.group(i)) for i in range(1, 5)]
        if all(0 <= o <= 255 for o in octets):
            matches.append(PIIMatch(
                text=m.group(),
                category="IP",
                start_char=m.start(),
                end_char=m.end(),
                source_block=block,
                confidence=1.0,
                detector="regex",
            ))
    return matches


# ---------------------------------------------------------------------------
# Credit cards  (Luhn-validated)
# ---------------------------------------------------------------------------

# 13–19 digit number, optionally grouped by spaces or hyphens
_CC_CANDIDATE_RE = re.compile(
    r"""
    (?<!\d)
    (?:\d[ \-]?){13,19}\d   # 13–19 digit groups, optional separators
    (?!\d)
    """,
    re.VERBOSE,
)


def _luhn_check(number_str: str) -> bool:
    """Return True if *number_str* (digits only) passes the Luhn checksum."""
    digits = [int(c) for c in number_str]
    digits.reverse()
    total = 0
    for i, d in enumerate(digits):
        if i % 2 == 1:
            d *= 2
            if d > 9:
                d -= 9
        total += d
    return total % 10 == 0


def detect_credit_cards(block: TextBlock) -> list[PIIMatch]:
    """
    Detect credit card numbers using candidate regex + Luhn checksum.

    Strips spaces/hyphens before Luhn validation.
    Rejects long financial reference numbers that do not pass Luhn.

    Positive: 4111 1111 1111 1111 (Luhn valid)
    Negative: 1234 5678 9012 3456 (Luhn invalid)
    """
    matches: list[PIIMatch] = []
    for m in _CC_CANDIDATE_RE.finditer(block.text):
        digits_only = re.sub(r"[ \-]", "", m.group())
        if 13 <= len(digits_only) <= 19 and _luhn_check(digits_only):
            matches.append(PIIMatch(
                text=m.group(),
                category="CREDIT_CARD",
                start_char=m.start(),
                end_char=m.end(),
                source_block=block,
                confidence=1.0,
                detector="regex",
            ))
    return matches


# ---------------------------------------------------------------------------
# Social Security Numbers (US format only)
# ---------------------------------------------------------------------------

_SSN_RE = re.compile(
    r"""
    (?<!\d)
    (?!000)(?!666)(?!9\d\d)  # invalid area numbers
    \d{3}
    -
    (?!00)\d{2}              # invalid group
    -
    (?!0000)\d{4}            # invalid serial
    (?!\d)
    """,
    re.VERBOSE,
)

# Context keywords that raise confidence (not required for detection)
_SSN_CONTEXT_KEYWORDS = re.compile(
    r"\b(?:SSN|Social Security(?: Number)?|Tax ID)\b",
    re.IGNORECASE,
)


def detect_ssns(block: TextBlock) -> list[PIIMatch]:
    """
    Detect US Social Security Numbers (XXX-XX-XXXX format).

    Rejects area 000/666/9xx, group 00, serial 0000.
    Context window of ±50 chars checked for SSN keywords to set confidence,
    but absence of keywords does NOT suppress detection.

    Note: PAN (AAAAA9999A) and Aadhaar (dddd dddd dddd) are explicitly
    outside the required scope and are NOT detected.

    Positive: 123-45-6789
    Negative: 000-00-0000, 123456789 (no hyphens), 900-45-6789
    """
    text = block.text
    matches: list[PIIMatch] = []
    for m in _SSN_RE.finditer(text):
        # Context window to adjust confidence
        window_start = max(0, m.start() - 50)
        window_end = min(len(text), m.end() + 50)
        context = text[window_start:window_end]
        confidence = 0.95 if _SSN_CONTEXT_KEYWORDS.search(context) else 0.80
        matches.append(PIIMatch(
            text=m.group(),
            category="SSN",
            start_char=m.start(),
            end_char=m.end(),
            source_block=block,
            confidence=confidence,
            detector="regex",
        ))
    return matches


# ---------------------------------------------------------------------------
# Dates — INTERNAL HELPER (not a required PII detector)
# ---------------------------------------------------------------------------

# Multi-format date patterns
_DATE_PATTERNS = [
    # DD/MM/YYYY or MM/DD/YYYY or YYYY/MM/DD with /, -, .
    re.compile(r"\b\d{1,2}[\/\-\.]\d{1,2}[\/\-\.]\d{2,4}\b"),
    # Month DD, YYYY  or  DD Month YYYY
    re.compile(
        r"\b(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|"
        r"Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|"
        r"Nov(?:ember)?|Dec(?:ember)?)\.?\s+\d{1,2},?\s+\d{4}\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b\d{1,2}\s+(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|"
        r"May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|"
        r"Nov(?:ember)?|Dec(?:ember)?)\.?\s+\d{4}\b",
        re.IGNORECASE,
    ),
    # ISO 8601: YYYY-MM-DD
    re.compile(r"\b\d{4}[-\/]\d{2}[-\/]\d{2}\b"),
]


def detect_dates(block: TextBlock) -> list[dict[str, Any]]:
    """
    INTERNAL HELPER — not called directly by the pipeline as a PII detector.

    Finds all date-shaped strings in block.text and returns raw dicts
    (not PIIMatch objects) for use by detect_dates_of_birth().

    Returns
    -------
    list of {"text": str, "start": int, "end": int, "format_hint": str}
    """
    text = block.text
    found: list[dict] = []
    seen_spans: set[tuple[int, int]] = set()

    for pattern in _DATE_PATTERNS:
        for m in pattern.finditer(text):
            span = (m.start(), m.end())
            if span in seen_spans:
                continue
            seen_spans.add(span)
            found.append({
                "text": m.group(),
                "start": m.start(),
                "end": m.end(),
            })

    return found


# ---------------------------------------------------------------------------
# Dates of Birth — requires context keyword
# ---------------------------------------------------------------------------

_DOB_CONTEXT_RE = re.compile(
    r"\b(?:DOB|D\.O\.B\.?|Date\s+of\s+Birth|Birth\s*Date|Birthdate|"
    r"Born|born\s+on)\b",
    re.IGNORECASE,
)

_DOB_CONTEXT_WINDOW = 100  # characters either side of candidate date


def detect_dates_of_birth(block: TextBlock) -> list[PIIMatch]:
    """
    Detect dates of birth by combining date candidates with a keyword context
    window.

    Only dates where a DOB keyword appears within ±100 characters are emitted.
    Plain filing dates, offer dates, incorporation dates, financial period
    dates — i.e. the vast majority of dates in a prospectus — are NOT emitted.

    Positive: "DOB: 15/03/1985"  →  match
    Negative: "Filed on 15/03/1985"  →  no match

    Confidence is 0.9 (heuristic).
    """
    text = block.text
    candidates = detect_dates(block)
    matches: list[PIIMatch] = []

    for cand in candidates:
        window_start = max(0, cand["start"] - _DOB_CONTEXT_WINDOW)
        window_end = min(len(text), cand["end"] + _DOB_CONTEXT_WINDOW)
        context = text[window_start:window_end]
        if _DOB_CONTEXT_RE.search(context):
            matches.append(PIIMatch(
                text=cand["text"],
                category="DOB",
                start_char=cand["start"],
                end_char=cand["end"],
                source_block=block,
                confidence=0.9,
                needs_review=False,
                detector="regex",
            ))

    return matches


# ---------------------------------------------------------------------------
# Convenience: run all regex detectors on a block
# ---------------------------------------------------------------------------

def run_all_regex_detectors(block: TextBlock) -> list[PIIMatch]:
    """
    Run every regex detector on *block* and return the combined list.
    Callers should pass results through resolver.resolve_overlaps().
    """
    results: list[PIIMatch] = []
    results.extend(detect_emails(block))
    results.extend(detect_phone_numbers(block))
    results.extend(detect_ip_addresses(block))
    results.extend(detect_credit_cards(block))
    results.extend(detect_ssns(block))
    results.extend(detect_dates_of_birth(block))
    return results
