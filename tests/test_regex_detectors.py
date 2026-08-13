"""
tests/test_regex_detectors.py
------------------------------
≥3 positive + ≥2 negative cases per detector.
DOB context-keyword gating is tested explicitly.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from models import TextBlock
from detectors.regex_detectors import (
    detect_emails,
    detect_phone_numbers,
    detect_ip_addresses,
    detect_credit_cards,
    detect_ssns,
    detect_dates,
    detect_dates_of_birth,
    _luhn_check,
)


def block(text):
    return TextBlock(text=text, location={"type": "paragraph", "para_idx": 0,
                                          "runs": [], "run_offsets": []})


# ---------------------------------------------------------------------------
# Email
# ---------------------------------------------------------------------------
class TestEmail:
    def test_simple(self):
        assert detect_emails(block("Send to cs.connect@kshinternational.com please"))
    def test_subdomain(self):
        assert detect_emails(block("user@mail.example.co.uk"))
    def test_plus_addressing(self):
        r = detect_emails(block("test+filter@example.com"))
        assert r[0].text == "test+filter@example.com"
    def test_no_at(self):
        assert not detect_emails(block("not_an_email"))
    def test_no_domain(self):
        assert not detect_emails(block("missing@"))
    def test_category(self):
        r = detect_emails(block("info@example.com"))
        assert r[0].category == "EMAIL"


# ---------------------------------------------------------------------------
# Phone numbers
# ---------------------------------------------------------------------------
class TestPhone:
    def test_international_indian(self):
        r = detect_phone_numbers(block("+91 20 4505 3237"))
        assert r
    def test_us_style(self):
        r = detect_phone_numbers(block("(555) 123-4567"))
        assert r
    def test_bare_indian_mobile(self):
        r = detect_phone_numbers(block("9876543210"))
        assert r
    def test_too_short(self):
        assert not detect_phone_numbers(block("123"))
    def test_too_long(self):
        assert not detect_phone_numbers(block("9999999999999999"))
    def test_category(self):
        r = detect_phone_numbers(block("+91 98765 43210"))
        assert r[0].category == "PHONE"


# ---------------------------------------------------------------------------
# IP addresses
# ---------------------------------------------------------------------------
class TestIP:
    def test_private_class_c(self):
        r = detect_ip_addresses(block("Server at 192.168.1.1"))
        assert r[0].text == "192.168.1.1"
    def test_loopback(self):
        r = detect_ip_addresses(block("127.0.0.1"))
        assert r
    def test_all_zeros(self):
        r = detect_ip_addresses(block("0.0.0.0"))
        assert r
    def test_out_of_range_octet(self):
        assert not detect_ip_addresses(block("999.999.999.999"))
    def test_out_of_range_one_octet(self):
        assert not detect_ip_addresses(block("256.0.0.1"))
    def test_category(self):
        r = detect_ip_addresses(block("10.0.0.1 is the gateway"))
        assert r[0].category == "IP"


# ---------------------------------------------------------------------------
# Credit cards (Luhn)
# ---------------------------------------------------------------------------
class TestCreditCard:
    def test_luhn_function_valid(self):
        assert _luhn_check("4111111111111111")  # Visa test number
    def test_luhn_function_invalid(self):
        assert not _luhn_check("1234567890123456")
    def test_detect_valid_visa(self):
        r = detect_credit_cards(block("Card: 4111 1111 1111 1111"))
        assert r
    def test_detect_valid_mastercard(self):
        r = detect_credit_cards(block("5500005555555559"))
        assert r
    def test_detect_valid_with_hyphens(self):
        r = detect_credit_cards(block("4111-1111-1111-1111"))
        assert r
    def test_reject_luhn_invalid(self):
        assert not detect_credit_cards(block("1234 5678 9012 3456"))
    def test_reject_short_number(self):
        assert not detect_credit_cards(block("123456789012"))


# ---------------------------------------------------------------------------
# SSN
# ---------------------------------------------------------------------------
class TestSSN:
    def test_standard(self):
        r = detect_ssns(block("SSN: 123-45-6789"))
        assert r[0].text == "123-45-6789"
    def test_high_confidence_with_keyword(self):
        r = detect_ssns(block("Social Security Number: 234-56-7890"))
        assert r[0].confidence > 0.9
    def test_lower_confidence_without_keyword(self):
        r = detect_ssns(block("Reference 234-56-7890"))
        assert r[0].confidence < 0.9
    def test_reject_all_zeros(self):
        assert not detect_ssns(block("000-00-0000"))
    def test_reject_no_hyphens(self):
        assert not detect_ssns(block("123456789"))
    def test_reject_invalid_area_666(self):
        assert not detect_ssns(block("666-12-3456"))
    def test_category(self):
        r = detect_ssns(block("SSN 523-45-6789"))
        assert r[0].category == "SSN"


# ---------------------------------------------------------------------------
# Dates helper
# ---------------------------------------------------------------------------
class TestDates:
    def test_slash_format(self):
        results = detect_dates(block("Date: 15/03/1985"))
        assert any("15/03/1985" in d["text"] for d in results)
    def test_iso(self):
        results = detect_dates(block("Filed: 2024-06-01"))
        assert any("2024-06-01" in d["text"] for d in results)
    def test_month_name(self):
        results = detect_dates(block("March 15, 1985"))
        assert results


# ---------------------------------------------------------------------------
# DOB — requires context keyword
# ---------------------------------------------------------------------------
class TestDOB:
    def test_with_dob_keyword(self):
        r = detect_dates_of_birth(block("DOB: 15/03/1985"))
        assert r
        assert r[0].category == "DOB"

    def test_with_born_keyword(self):
        r = detect_dates_of_birth(block("Born: 01 January 1990"))
        assert r

    def test_with_date_of_birth_phrase(self):
        r = detect_dates_of_birth(block("Date of Birth: 22/07/1978"))
        assert r

    def test_plain_filing_date_not_emitted(self):
        r = detect_dates_of_birth(block("Filed on 15/03/1985 with the authority"))
        assert not r, "Plain filing date should not be classified as DOB"

    def test_offer_date_not_emitted(self):
        r = detect_dates_of_birth(block("Offer opens: 01/04/2024"))
        assert not r, "Offer date should not be classified as DOB"

    def test_confidence_is_heuristic(self):
        r = detect_dates_of_birth(block("DOB 10/05/1995"))
        assert r[0].confidence == pytest.approx(0.9)
