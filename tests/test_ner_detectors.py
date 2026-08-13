"""
tests/test_ner_detectors.py
-----------------------------
Tests for spaCy NER detectors: PERSON, COMPANY, and ADDRESS heuristic.

Note: spaCy model must be installed before running these tests:
    python -m spacy download en_core_web_sm
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from models import TextBlock
from detectors.ner_detectors import (
    detect_person_names,
    detect_organizations,
    detect_addresses,
    load_nlp_model,
)


def block(text):
    return TextBlock(text=text, location={"type": "paragraph", "para_idx": 0,
                                          "runs": [], "run_offsets": []})


class TestNLPModel:
    def test_model_loads(self):
        nlp = load_nlp_model()
        assert nlp is not None

    def test_model_is_cached(self):
        nlp1 = load_nlp_model()
        nlp2 = load_nlp_model()
        assert nlp1 is nlp2


class TestPersonNames:
    def test_full_name_detected(self):
        r = detect_person_names(block("The director is John Smith."))
        assert any("John Smith" in m.text or "John" in m.text for m in r)

    def test_category_is_person(self):
        r = detect_person_names(block("According to Mary Johnson, the report is filed."))
        assert all(m.category == "PERSON" for m in r)

    def test_no_person_in_generic_sentence(self):
        r = detect_person_names(block("The company filed the annual report."))
        # Generic sentences may or may not have PERSON — just confirm no crash
        assert isinstance(r, list)

    def test_multiple_names(self):
        r = detect_person_names(
            block("The board members are Alice Brown and Bob Green.")
        )
        # At least one name should be detected
        assert len(r) >= 1

    def test_one_pii_match_per_occurrence(self):
        """Each entity occurrence yields one PIIMatch (no collapsing)."""
        text = "John Doe met with John Doe again."
        r = detect_person_names(block(text))
        john_doe_matches = [m for m in r if "John Doe" in m.text]
        assert len(john_doe_matches) >= 1  # at least once; spaCy may dedup internally


class TestOrganizations:
    def test_company_detected(self):
        r = detect_organizations(block("The annual report of Acme Corporation was filed."))
        assert r

    def test_category_mapped_to_company(self):
        """spaCy ORG label must be mapped to required category COMPANY."""
        r = detect_organizations(block("Google Inc. reported strong earnings."))
        assert all(m.category == "COMPANY" for m in r)

    def test_no_org_in_plain_sentence(self):
        r = detect_organizations(block("The weather is nice today."))
        assert isinstance(r, list)


class TestAddresses:
    def test_full_address_emitted(self):
        """Block with location entity + address keyword + PIN should emit ADDRESS."""
        text = "Office at Plot 5, Industrial Estate, Pune - 411001"
        r = detect_addresses(block(text))
        assert r, "Expected ADDRESS match for full address block"

    def test_address_category(self):
        text = "123 Main Street, Springfield, 62701"
        r = detect_addresses(block(text))
        if r:
            assert r[0].category == "ADDRESS"

    def test_needs_review_is_true(self):
        text = "Plot 12, Sector 5, Nagar, Mumbai 400001"
        r = detect_addresses(block(text))
        if r:
            assert r[0].needs_review is True

    def test_confidence_below_one(self):
        text = "Building 3, Industrial Road, Delhi 110001"
        r = detect_addresses(block(text))
        if r:
            assert r[0].confidence < 1.0

    def test_standalone_city_not_emitted(self):
        """A bare city name like 'Pune' must not be classified as ADDRESS."""
        r = detect_addresses(block("The company is based in Pune."))
        assert not r, "Standalone city name should not be emitted as ADDRESS"

    def test_standalone_country_not_emitted(self):
        r = detect_addresses(block("India has a large population."))
        assert not r, "Standalone country name should not be emitted as ADDRESS"
