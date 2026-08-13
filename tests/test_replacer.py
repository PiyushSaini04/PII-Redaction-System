"""
tests/test_replacer.py
-----------------------
Tests for build_mapping() and _generate_replacement():
  - Same seed → identical mapping across two independent calls
  - Same original text called twice → same fake value (consistency)
  - EMAIL fake passes detect_emails pattern
  - IP fake is in RFC 5737 documentation range
  - SSN fake matches XXX-XX-XXXX with valid area/group/serial
  - CREDIT_CARD fake passes Luhn check
  - DOB format is preserved
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import re
import pytest
from models import TextBlock, PIIMatch
from replacer import build_mapping, _generate_replacement
from detectors.regex_detectors import detect_emails, detect_ip_addresses, _luhn_check


# Re-expose _luhn_check via replacer for convenience test
def _luhn_check_str(s: str) -> bool:
    digits = re.sub(r"[ \-]", "", s)
    return _luhn_check(digits)


def _block():
    return TextBlock(text="", location={"type": "paragraph", "para_idx": 0,
                                        "runs": [], "run_offsets": []})


def _make_matches(items):
    """items: list of (text, category)"""
    b = _block()
    return [
        PIIMatch(text=t, category=c, start_char=0, end_char=len(t), source_block=b)
        for t, c in items
    ]


class TestBuildMapping:
    def test_same_seed_same_mapping(self):
        matches = _make_matches([
            ("John Smith", "PERSON"),
            ("john@example.com", "EMAIL"),
        ])
        m1 = build_mapping(matches, seed=42)
        m2 = build_mapping(matches, seed=42)
        assert m1 == m2

    def test_different_seeds_different_mapping(self):
        matches = _make_matches([("John Smith", "PERSON")])
        m1 = build_mapping(matches, seed=1)
        m2 = build_mapping(matches, seed=99)
        # Very unlikely to collide
        assert m1.get("John Smith") != m2.get("John Smith")

    def test_same_text_maps_to_same_fake(self):
        matches = _make_matches([
            ("John Smith", "PERSON"),
            ("John Smith", "PERSON"),  # duplicate
        ])
        mapping = build_mapping(matches, seed=42)
        assert "John Smith" in mapping
        # Only one entry per unique text
        assert len(mapping) == 1

    def test_multiple_uniques(self):
        matches = _make_matches([
            ("Alice Brown", "PERSON"),
            ("Bob Green", "PERSON"),
            ("carol@test.com", "EMAIL"),
        ])
        mapping = build_mapping(matches, seed=42)
        assert len(mapping) == 3
        assert "Alice Brown" in mapping
        assert "Bob Green" in mapping
        assert "carol@test.com" in mapping

    def test_email_fake_passes_email_pattern(self):
        matches = _make_matches([("real@domain.com", "EMAIL")])
        mapping = build_mapping(matches, seed=42)
        fake_email = mapping["real@domain.com"]
        b = TextBlock(text=fake_email, location={"type": "paragraph", "para_idx": 0,
                                                  "runs": [], "run_offsets": []})
        assert detect_emails(b), f"Fake email '{fake_email}' failed email detection"

    def test_ip_fake_is_documentation_range(self):
        matches = _make_matches([("10.0.0.1", "IP")])
        mapping = build_mapping(matches, seed=42)
        fake_ip = mapping["10.0.0.1"]
        assert fake_ip.startswith("192.0.2.") or fake_ip.startswith("198.51.100."), \
            f"Fake IP '{fake_ip}' not in RFC 5737 documentation range"

    def test_ssn_fake_matches_format(self):
        matches = _make_matches([("123-45-6789", "SSN")])
        mapping = build_mapping(matches, seed=42)
        fake_ssn = mapping["123-45-6789"]
        assert re.match(
            r"(?!000)(?!666)(?!9\d\d)\d{3}-(?!00)\d{2}-(?!0000)\d{4}$",
            fake_ssn
        ), f"Fake SSN '{fake_ssn}' failed format check"

    def test_credit_card_fake_passes_luhn(self):
        matches = _make_matches([("4111111111111111", "CREDIT_CARD")])
        mapping = build_mapping(matches, seed=42)
        fake_cc = mapping["4111111111111111"]
        assert _luhn_check_str(fake_cc), f"Fake CC '{fake_cc}' failed Luhn"

    def test_dob_preserves_slash_format(self):
        matches = _make_matches([("15/03/1985", "DOB")])
        mapping = build_mapping(matches, seed=42)
        fake_dob = mapping["15/03/1985"]
        assert re.match(r"\d{2}/\d{2}/\d{4}", fake_dob), \
            f"Fake DOB '{fake_dob}' did not preserve DD/MM/YYYY format"

    def test_indian_phone_preserves_plus91(self):
        matches = _make_matches([("+91 98765 43210", "PHONE")])
        mapping = build_mapping(matches, seed=42)
        fake_phone = mapping["+91 98765 43210"]
        assert fake_phone.startswith("+91"), \
            f"Indian phone replacement '{fake_phone}' did not preserve +91 prefix"
