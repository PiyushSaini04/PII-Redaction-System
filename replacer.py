"""
replacer.py
-----------
Part A — Canonical fake-value mapping
    build_mapping(matches, seed)    -> dict[str, str]
    get_replacement(text, category, mapping, fake) -> str

Part B — DOCX in-place rewriting
    apply_to_docx(doc, matches, mapping) -> None

Design principles
-----------------
* Consistency:  the same original PII text always maps to the same fake value
  throughout the entire document, regardless of how many times it appears.
* Format-preservation:  replacements look structurally similar to the original
  (e.g. +91 phone stays +91-prefixed, SSN stays XXX-XX-XXXX, email uses a
  safe synthetic domain).
* Determinism:  given the same seed, build_mapping() always produces the same
  mapping for the same set of matches.
* Safety:  IP replacements use RFC 5737 documentation ranges (192.0.2.x,
  198.51.100.x); email domains use example.com/testmail.org; SSNs avoid
  invalid area/group/serial combinations.
"""

from __future__ import annotations

import random
import re
from typing import Optional

from faker import Faker

from docx_io import apply_replacement_to_block
from models import PIIMatch, TextBlock


# ---------------------------------------------------------------------------
# Category-specific fake generators
# ---------------------------------------------------------------------------

def _fake_email(original: str, fake: Faker) -> str:
    """
    Generate a plausible but synthetic email.
    Preserves user@domain.tld structure; uses safe synthetic domain.
    """
    user = fake.user_name()
    safe_domains = ["example.com", "testmail.org", "sample.net", "demo.org"]
    domain = fake.random_element(safe_domains)
    return f"{user}@{domain}"


def _fake_phone(original: str, fake: Faker) -> str:
    """
    Generate a synthetic phone number that preserves country-code structure.
    An Indian +91 number stays +91-prefixed.
    """
    stripped = re.sub(r"[\s\-\(\)]", "", original)
    if stripped.startswith("+91") or (original.startswith("+91")):
        # Indian format: +91 XXXXX XXXXX
        digits = "".join([str(random.randint(6, 9))] +
                         [str(random.randint(0, 9)) for _ in range(9)])
        return f"+91 {digits[:5]} {digits[5:]}"
    elif stripped.startswith("+"):
        # Generic international
        cc = re.match(r"\+\d{1,3}", original)
        cc_str = cc.group() if cc else "+1"
        local = " ".join(
            "".join(str(random.randint(0, 9)) for _ in range(4))
            for _ in range(2)
        )
        return f"{cc_str} {local}"
    else:
        # US-style or bare 10-digit
        return fake.numerify("(###) ###-####")


def _fake_ssn(fake: Faker) -> str:
    """
    Generate a synthetic US SSN (XXX-XX-XXXX).
    Avoids invalid area numbers (000, 666, 900-999), group 00, serial 0000.
    """
    while True:
        area = random.randint(1, 899)
        if area == 666:
            continue
        group = random.randint(1, 99)
        serial = random.randint(1, 9999)
        candidate = f"{area:03d}-{group:02d}-{serial:04d}"
        # double-check the SSN regex would accept it
        if re.match(r"(?!000)(?!666)(?!9\d\d)\d{3}-(?!00)\d{2}-(?!0000)\d{4}$",
                    candidate):
            return candidate


def _fake_credit_card(fake: Faker) -> str:
    """Generate a Luhn-valid synthetic credit card number."""
    return fake.credit_card_number(card_type=None)


def _fake_dob(original: str, fake: Faker) -> str:
    """
    Generate a synthetic date of birth preserving the original date format.
    """
    from faker.providers.date_time import Provider as DateProvider
    import datetime

    dob = fake.date_of_birth(minimum_age=18, maximum_age=90)

    # Detect original format and match it
    if re.match(r"\d{4}-\d{2}-\d{2}", original):
        return dob.strftime("%Y-%m-%d")
    elif re.match(r"\d{1,2}/\d{1,2}/\d{4}", original):
        return dob.strftime("%d/%m/%Y")
    elif re.match(r"\d{1,2}-\d{1,2}-\d{4}", original):
        return dob.strftime("%d-%m-%Y")
    elif re.match(r"\d{1,2}\.\d{1,2}\.\d{4}", original):
        return dob.strftime("%d.%m.%Y")
    elif re.search(r"[A-Za-z]", original):
        # Month name present
        return dob.strftime("%B %d, %Y")
    else:
        return dob.strftime("%d/%m/%Y")


def _fake_ip(fake: Faker) -> str:
    """
    Generate a synthetic IP address from RFC 5737 documentation ranges:
      192.0.2.0/24  or  198.51.100.0/24
    These are reserved for documentation/testing and will never route.
    """
    prefix = fake.random_element(["192.0.2", "198.51.100"])
    last_octet = random.randint(1, 254)
    return f"{prefix}.{last_octet}"


# ---------------------------------------------------------------------------
# Part A — Canonical mapping builder
# ---------------------------------------------------------------------------

def build_mapping(
    matches: list[PIIMatch],
    seed: int = 42,
) -> dict[str, str]:
    """
    Build a dict[original_text -> fake_value] for every unique PII text value.

    The same original value always maps to the same fake value regardless of
    how many occurrences exist.  Multiple occurrences are handled by looking
    up the mapping rather than regenerating.

    Consistency example
    -------------------
    "Kushal Subbayya Hegde" appears 10 times → one entry in the mapping,
    all 10 PIIMatch occurrences replaced with the same fake value.

    Determinism
    -----------
    Faker and random are both seeded with *seed* before any generation.
    Same matches + same seed → identical mapping across independent runs.
    """
    fake = Faker()
    Faker.seed(seed)
    random.seed(seed)

    mapping: dict[str, str] = {}

    for match in matches:
        if match.text in mapping:
            continue  # already assigned — consistency guaranteed

        replacement = _generate_replacement(match.text, match.category, fake)
        mapping[match.text] = replacement

    return mapping


def _generate_replacement(text: str, category: str, fake: Faker) -> str:
    """
    Generate one synthetic replacement for a given (text, category) pair.
    Called only once per unique text value.
    """
    cat = category.upper()

    if cat == "PERSON":
        return fake.name()
    elif cat == "EMAIL":
        return _fake_email(text, fake)
    elif cat == "PHONE":
        return _fake_phone(text, fake)
    elif cat == "COMPANY":
        return fake.company()
    elif cat == "ADDRESS":
        # Generate a plausible street address
        return f"{fake.building_number()} {fake.street_name()}, {fake.city()}"
    elif cat == "SSN":
        return _fake_ssn(fake)
    elif cat == "CREDIT_CARD":
        return _fake_credit_card(fake)
    elif cat == "DOB":
        return _fake_dob(text, fake)
    elif cat == "IP":
        return _fake_ip(fake)
    else:
        # Unknown category — return a generic placeholder
        return f"[REDACTED_{cat}]"


def get_replacement(
    text: str,
    category: str,
    mapping: dict[str, str],
    fake: Faker,
) -> str:
    """
    Return mapping[text] if it exists (consistency guarantee).
    If not yet mapped, generate, store, and return.
    Used for incremental mapping during streaming pipelines.
    """
    if text not in mapping:
        mapping[text] = _generate_replacement(text, category, fake)
    return mapping[text]


# ---------------------------------------------------------------------------
# Part B — DOCX in-place rewriting
# ---------------------------------------------------------------------------

def apply_to_docx(
    matches: list[PIIMatch],
    mapping: dict[str, str],
) -> None:
    """
    Apply the replacement mapping to the DOCX by modifying Run objects in place.

    Processing order
    ----------------
    Matches are processed from LAST to FIRST (descending start_char) within
    each block.  This ensures that replacing a later span does not shift the
    character offsets of earlier spans in the same block.

    Multi-run spans
    ---------------
    Delegated to docx_io.apply_replacement_to_block(), which handles PII
    that is split across multiple Word runs due to formatting boundaries
    (e.g. part of a name is bold).

    Coverage
    --------
    All block types are handled: paragraphs, table cells, headers, footers.
    The block type is transparent to this function — it operates on the
    TextBlock's run list regardless of source.

    Parameters
    ----------
    matches : All resolved PIIMatch occurrences (from resolver.collect_all_matches).
    mapping : dict from build_mapping(); original_text -> fake_value.
    """
    # Group matches by their source block (using object identity)
    from collections import defaultdict
    block_matches: dict[int, list[PIIMatch]] = defaultdict(list)

    for match in matches:
        block_id = id(match.source_block)
        block_matches[block_id].append(match)

    # Process each block: reverse sort by start_char to preserve offsets
    for block_id, bmatches in block_matches.items():
        # Sort descending by start_char
        bmatches_sorted = sorted(bmatches, key=lambda m: m.start_char, reverse=True)

        for match in bmatches_sorted:
            replacement = mapping.get(match.text)
            if replacement is None:
                continue  # safety: skip unmapped (shouldn't happen)

            apply_replacement_to_block(
                block=match.source_block,
                start_char=match.start_char,
                end_char=match.end_char,
                replacement=replacement,
            )
