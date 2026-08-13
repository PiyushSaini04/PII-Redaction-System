"""
detectors/ner_detectors.py
--------------------------
spaCy NER-based detectors for unstructured PII.

Architecture: spaCy entities are CANDIDATES only.
Every PERSON and ORG entity passes through a structured validation layer
(_validate_person / _validate_company) before becoming a PIIMatch.

Key design decision for COMPANY validation
------------------------------------------
The validator uses a REQUIRE-POSITIVE-EVIDENCE policy: an entity is
accepted ONLY if it exhibits explicit structural markers of a real
organisation (company suffix, named-trust pattern, known public body).
Everything else is rejected by default. This is the critical difference
from the previous accept-by-default fallback that caused false positives
such as "Non-GAAP Measures", "Freehold Land and Leasehold Land", etc.

Entity-type taxonomy
--------------------
T1 Private companies    — end in Limited / Ltd / LLP / LLC / Inc / GmbH …
T2 Named trusts         — <PROPER NAME> FAMILY TRUST / FOUNDATION / TRUST
T3 Public bodies        — SEBI, RBI, Securities and Exchange Board of India …
T4 Professional firms   — <Name> & <Name> LLP/Ltd (suffix required)
T5 Person names         — ≥2 name tokens, letter-only, not a role/location
T6 Financial/legal FP   — generic terms, IPO mechanics, section headings
T7 Clause fragments     — ALL-CAPS multi-word legal extracts

Family-trust redaction policy
------------------------------
Named promoter trusts (DHAULAGIRI FAMILY TRUST etc.) match T2 and were
accepted in earlier iterations. Based on user feedback the trusts are
NOT redacted — they are added to the company FP blocklist.
"""

from __future__ import annotations

import functools
import re
from dataclasses import dataclass

import spacy
from spacy.language import Language

from models import PIIMatch, TextBlock


# ---------------------------------------------------------------------------
# Cached model loader
# ---------------------------------------------------------------------------

@functools.lru_cache(maxsize=1)
def load_nlp_model() -> Language:
    try:
        return spacy.load("en_core_web_sm")
    except OSError as exc:
        raise RuntimeError(
            "spaCy model 'en_core_web_sm' not found.  "
            "Run: python -m spacy download en_core_web_sm"
        ) from exc


# ---------------------------------------------------------------------------
# ValidationResult
# ---------------------------------------------------------------------------

@dataclass
class ValidationResult:
    accepted:           bool
    confidence:         float
    needs_review:       bool
    reason:             str        # machine-readable code
    suggested_category: str


# ---------------------------------------------------------------------------
# Patterns
# ---------------------------------------------------------------------------

# T1 — company-type suffixes (anchored to end of string)
_COMPANY_SUFFIX_RE = re.compile(
    r"\b(?:private\s+limited|pvt\.?\s+ltd\.?|limited|ltd\.?|llp|llc|inc\.?|"
    r"corporation|corp\.?|incorporated|plc|gmbh|bv|nv)\s*$",
    re.IGNORECASE,
)

# T2 — named trust / foundation: needs a substantive prefix (≥1 proper word)
_NAMED_TRUST_RE = re.compile(
    r"^[A-Za-z][A-Za-z0-9\s\-]+\s+(?:FAMILY\s+TRUST|FOUNDATION|CHARITABLE\s+TRUST)$",
    re.IGNORECASE,
)

# Extra org-noun suffix that should redirect PERSON entities to COMPANY pipeline
# (then COMPANY pipeline will reject most of them via no_positive_evidence)
_ORG_NOUN_SUFFIX_RE = re.compile(
    r"\b(?:company|group|bank|fund|authority|commission|agency|department|"
    r"ministry|council|institute|association|society|cooperative|ventures?|"
    r"authority|bureau|board|trust|foundation)\s*$",
    re.IGNORECASE,
)

# Act / Regulation names — reject even when they have "Limited" in middle
_ACT_REGULATION_RE = re.compile(
    r"\b(?:act|regulations?|regulatory|rules|ordinance|statute|code|bill)\b",
    re.IGNORECASE,
)

# T3 — known public / regulatory bodies (accept with needs_review)
_PUBLIC_BODY_SET: frozenset[str] = frozenset({
    "registrar of companies",
    "national stock exchange of india",
    "national stock exchange",
    "bse limited",
    "reserve bank of india",
    "insurance regulatory and development authority of india",
    "pension fund regulatory and development authority",
    "sebi", "nse", "bse", "rbi", "irda", "pfrda",
})

# IPO/offer mechanics pattern — reject these as COMPANY
_IPO_MECHANISM_RE = re.compile(
    r"^(?:the\s+)?[\w\s\-]*"
    r"(?:portion|price|period|shares|investors|buyers|funds|banks|"
    r"proceeds|allotment|tranche|category|bidders|intermediaries)$",
    re.IGNORECASE,
)

# Pattern for a plausible person name (letters, spaces, hyphens, dots, apostrophes)
_NAME_CHAR_RE = re.compile(r"^[a-zA-Z\s\-\.\'\u00C0-\u024F]+$", re.UNICODE)

# Trailing footnote / table markers to strip from entity text
_FOOTNOTE_STRIP_RE = re.compile(r"[\*\^\#\&@\u2019\u2018\u201c\u201d]+$")


# ---------------------------------------------------------------------------
# PERSON FP blocklist  (case-insensitive, exact lowercase match)
# ---------------------------------------------------------------------------

_PERSON_FP_BLOCKLIST: frozenset[str] = frozenset({
    # IPO roles and shorthand
    "offer", "promoters", "promoter", "directors", "director",
    "bidders", "bidding", "bids", "bid", "shareholders", "shareholder",
    "company", "registrar", "auditor", "pre-offer",
    "post-offer", "post-offer equity share", "offer for",
    "anchor investors", "upi bidders", "mutual funds", "syndicate",
    "investor", "investors", "underwriter", "selling shareholder",
    "executive directors", "statutory auditors",
    # Price and metric terms
    "cap price", "floor price", "reference rate", "offer price", "bid price",
    "issue price", "face value", "cut-off price",
    "pat cagr", "ebitda cagr", "revenue cagr",
    "net proceeds", "gross proceeds", "bonus",
    # Financial reporting / accounting terms
    "non-gaap measures", "non-gaap", "gaap",
    # Legal / document
    "prospectus", "red herring prospectus", "email", "e-mail", "website",
    "corrigenda thereto", "key managerial", "key managerial personnel",
    # Section heading patterns
    "b.  non-gaap measures", "c.  operational",
    # Single-word misc FP
    "bid", "vikhroli", "nuvama", "baner", "chakan", "rohit branch",
    "risks", "brlm", "sebi", "nse", "bse", "ipo",
    "qib", "hni", "rii", "asba",
    "widely circulated marathi daily newspaper",
    "secondary transfer of",
    # Location fragments that spaCy tags as PERSON
    "supa facility", "chakan taluka - khed", "bandra east", "taluka khed",
    "marg backbay reclamation churchgate", "deccan gymkhana",
    "waterloo industrial", "waterloo industrial park",
    "bandra kurla complex", "bkc",
})


# ---------------------------------------------------------------------------
# COMPANY FP blocklist  (case-insensitive, exact lowercase match)
# ---------------------------------------------------------------------------

_COMPANY_FP_BLOCKLIST: frozenset[str] = frozenset({
    # Corporate governance generic nouns
    "board", "company", "promoters", "directors", "management",
    "audit committee", "key managerial personnel", "independent directors",
    "the executive directors", "statutory auditors",
    "audit committee, board", "the board of directors",
    "board of directors", "the board and shareholders",
    "board and shareholders",
    "securities and exchange board of india",
    "the securities and exchange board of india",
    "securities and exchange board of india act",
    "rohit branch", "post-offer", "post-offer equity share", "offer for",
    # IPO mechanics — these are generic instrument / process terms
    "offer", "the offer", "equity", "equity shares", "bids", "bidders",
    "anchor investors", "the offer price", "the offered shares",
    "the bid/offer period", "the offer for sale",
    "the promoter selling shareholders",
    "the non-institutional portion", "the net qib portion",
    "the designated stock exchange", "the net proceeds",
    "the qib portion", "non-institutional portion",
    "qualified institutional buyers", "retail individual investors",
    "the sponsor banks", "upi bidders",
    "promoter selling shareholders", "promoter selling", "equity share",
    "the qib portion to anchor investors", "split of equity shares",
    "allotment", "offer price", "bonus", "bidding",
    "the bid/offer opening date", "the bid/offer closing date",
    "the bid amount", "non-institutional investors and retail individual investors",
    # Document / regulation titles
    "red herring prospectus", "prospectus", "draft red herring prospectus",
    "the restated financial statements",
    "the securities contracts (regulation) rules",
    "icdr master circular", "the care report", "care report",
    "the general information document", "offer related terms",
    "general terms and abbreviations", "draft prospectus", "the prospectus",
    "market data and currency of presentation",
    "market data and currency of",
    "the stock exchanges for the offer",
    "national stock exchange of",    # truncated reference
    # Section headings / acronyms
    "definitions", "currency", "issuer", "internal risks", "external risks",
    "e-mail", "upi id", "cogs", "scra", "fema", "icai",
    "asba", "ipo", "risks",
    # Generic financial and infrastructure terms
    "registrar", "registered office", "corporate office", "inter alia",
    "syndicate", "mutual funds", "non-institutional investors",
    "designated intermediaries", "promoter group", "up", "maharashtra",
    "life insurance companies and pension funds",
    "the life insurance companies and pension funds",
    "mutual funds, life insurance companies and pension funds",
    # All-caps clause fragments
    "collectively", "the offer shall constitute",
    "equity share capital of our company", "sale of", "sale",
    "family trust", "bid/offer period",
    # Named trusts — per user policy, promoter trusts are NOT redacted
    "dhaulagiri family trust", "makalu family trust",
    "broad family trust", "annapurna family trust",
    "kanchenjunga family trust", "everest family trust",
    # IPO mechanics continued
    "bid/offer closing day", "bid/offer opening day",
    # Balance sheet / financial statement line items
    "freehold land and leasehold land", "bank balances and advances",
    "employer and employee contribution",
    "non-gaap measures",
    # Infrastructure
    "tower 2a & 2b",
})


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_PERSON_TOKEN_BLOCKLIST = frozenset({
    "amount", "margin", "conditioning", "hospital", "chambers", "slip", "account",
    "broker", "kilometers", "amperes", "volts", "website", "individual", "trusts",
    "promoter", "bidders", "transfer", "listing", "schedule", "defaulter", "showroom",
    "park", "facility", "complex", "road", "lane", "street", "building", "court",
    "apartment", "colony", "society", "reconciliation", "measures", "operational",
    "key", "personnel", "auditor", "auditors", "registrar", "company", "group", "bank",
    "fund", "trust", "foundation", "limited", "ltd", "llp", "inc", "corp", "board",
    "committee", "management", "directors", "promoters", "shareholders", "shareholder",
    "officer", "officers", "secretary", "advisors", "advisor", "consultant", "consultants",
    "valuation", "value", "price", "cagr", "ebitda", "revenue", "proceeds", "bonus",
    "bid", "offer", "issue", "allotment", "underwriter", "syndicate", "member", "members",
    "date", "period", "closing", "opening", "day", "year", "month", "week",
    "acknowledgement", "air", "voltaic", "photo", "mega", "east", "west", "north", "south",
    "central", "state", "government", "ministry", "department", "authority", "commission",
    "agency", "council", "institute", "cooperative", "association", "university", "college",
    "school", "pune", "mumbai", "india", "maharashtra", "delhi", "bengaluru", "chennai",
    "kolkata", "hyderabad", "electricals", "industries", "engineering", "metals",
    "extrusion", "technologies", "software", "systems", "solutions", "gram", "kisan",
    "urja", "suraksha"
})

_COMPANY_SUFFIX_WORDS = frozenset({
    "private", "pvt", "pvt.", "limited", "ltd", "ltd.", "llp", "llc", "inc", "inc.",
    "corporation", "corp", "corp.", "incorporated", "company", "co", "co.", "bank",
    "trust", "group", "fund", "committee", "board", "firm", "foundation"
})

def _clean_entity_text(text: str) -> str:
    """Strip trailing footnote / table annotation markers."""
    return _FOOTNOTE_STRIP_RE.sub("", text).strip()


def _get_preceding_context(doc, ent, window: int = 5) -> str:
    start = max(0, ent.start - window)
    return doc[start:ent.start].text.lower()


# ---------------------------------------------------------------------------
# Validation functions
# ---------------------------------------------------------------------------

def _validate_person(raw_text: str, block_text: str,
                     preceding_context: str) -> ValidationResult:
    """
    Return a ValidationResult for a spaCy PERSON candidate.
    Uses a reject-unless-evidence-present policy for common FP patterns,
    then accepts multi-token name-like strings.
    """
    text_stripped = _clean_entity_text(raw_text)
    text_lower = text_stripped.lower()

    # 0. Company-suffix conflict resolution (runs before all PERSON checks)
    if _COMPANY_SUFFIX_RE.search(text_stripped):
        return _validate_company(text_stripped, block_text, preceding_context)
    if _ORG_NOUN_SUFFIX_RE.search(text_stripped):
        # redirect; COMPANY validator will almost certainly reject via
        # no_positive_evidence — but that is the correct outcome
        return _validate_company(text_stripped, block_text, preceding_context)

    # 1. Exact blocklist match
    if text_lower in _PERSON_FP_BLOCKLIST:
        return ValidationResult(False, 0.0, False, "blocklist_match", "PERSON")

    # 2. Role-noun suffix
    if re.search(r"\b(?:branch|agents|personnel|committee|shareholders|"
                 r"auditors|managers|executives)\s*$", text_lower):
        return ValidationResult(False, 0.0, False, "role_noun_suffix", "PERSON")

    # 3. Location-indicator words
    if re.search(r"\b(?:taluka|gymkhana|marg|reclamation|churchgate|"
                 r"nagar|sector|village|district)\b", text_lower):
        return ValidationResult(False, 0.0, False, "location_pattern", "PERSON")

    # 4. Financial / accounting metric abbreviations (ALL-CAPS pairs)
    if re.match(r"^[A-Z]+\s+[A-Z]+$", text_stripped):
        return ValidationResult(False, 0.0, False, "allcaps_metric", "PERSON")

    # 5. Section-heading prefix (e.g. "B.  Non-GAAP Measures")
    if re.match(r"^[A-Z]\.\s+", text_stripped):
        return ValidationResult(False, 0.0, False, "section_heading", "PERSON")

    # 6. Character-pattern check — must be name-like (no digits, colons, slashes)
    if not _NAME_CHAR_RE.match(text_stripped):
        return ValidationResult(False, 0.0, False, "special_chars", "PERSON")

    tokens = text_stripped.split()

    # 7. Token blocklist check — reject if any token is a known generic/business word
    if any(t.lower() in _PERSON_TOKEN_BLOCKLIST for t in tokens):
        return ValidationResult(False, 0.0, False, "token_blocklist", "PERSON")

    # Evidence aggregation
    confidence = 0.80
    needs_review = False

    has_context = any(kw in preceding_context for kw in (
        "signed by", "director", "appointed", "represented by", "contact", "named",
        "mr.", "ms.", "mrs.", "dr.", "mr", "ms", "mrs", "dr", "promoter", "shri", "sh.", "sh"
    ))

    if len(tokens) < 2:
        if not has_context:
            return ValidationResult(False, 0.0, False, "single_token_without_context", "PERSON")
        confidence = 0.65
        needs_review = True

    # Name-introducing context boosts confidence
    if has_context:
        confidence = min(1.0, confidence + 0.10)

    # All-caps name: plausible in prospectus tables (e.g. KUSHAL SUBBAYYA HEGDE)
    if text_stripped.isupper():
        confidence = min(confidence, 0.75)
        needs_review = True

    if confidence >= 0.50:
        return ValidationResult(True, confidence, needs_review, "accepted", "PERSON")
    return ValidationResult(False, confidence, False, "low_confidence", "PERSON")


def _validate_company(raw_text: str, block_text: str,
                      preceding_context: str) -> ValidationResult:
    """
    Return a ValidationResult for a spaCy ORG candidate.

    REQUIRE-POSITIVE-EVIDENCE policy: an entity is accepted ONLY when it
    exhibits explicit structural evidence of being a real organisation.
    Anything that does not match T1/T2/T3 is rejected by default.
    This eliminates the accept-by-default fallback that caused fabrications
    such as 'Non-GAAP Measures' being replaced by a Faker company name.
    """
    text_stripped = _clean_entity_text(raw_text)
    text_lower = text_stripped.lower()

    if not text_stripped:
        return ValidationResult(False, 0.0, False, "empty_after_cleaning", "COMPANY")

    # 1. Exact blocklist match — immediate reject
    if text_lower in _COMPANY_FP_BLOCKLIST:
        return ValidationResult(False, 0.0, False, "blocklist_match", "COMPANY")

    # 2. Act / Regulation name — reject even if it contains "Limited" somewhere
    if _ACT_REGULATION_RE.search(text_stripped):
        return ValidationResult(False, 0.0, False, "act_regulation_name", "COMPANY")

    # 3. T1 — company suffix: accept
    if _COMPANY_SUFFIX_RE.search(text_stripped):
        tokens = text_stripped.split()
        if all(t.lower() in _COMPANY_SUFFIX_WORDS for t in tokens):
            return ValidationResult(False, 0.0, False, "standalone_suffix", "COMPANY")
        return ValidationResult(True, 0.90, False, "company_suffix", "COMPANY")

    # 4. T2 — named trust / foundation: accept
    if _NAMED_TRUST_RE.match(text_stripped):
        return ValidationResult(True, 0.85, False, "named_trust", "COMPANY")

    # 5. T3 — known public / regulatory body: accept with needs_review
    if text_lower in _PUBLIC_BODY_SET:
        return ValidationResult(True, 0.60, True, "public_body", "COMPANY")

    # 6. All-caps clause fragment (T7) — three or more all-caps tokens
    tokens = text_stripped.split()
    if text_stripped.isupper() and len(tokens) >= 3:
        return ValidationResult(False, 0.0, False, "allcaps_clause_fragment", "COMPANY")

    # 7. IPO / financial mechanism phrase — structural pattern
    if _IPO_MECHANISM_RE.match(text_stripped):
        return ValidationResult(False, 0.0, False, "ipo_mechanism_pattern", "COMPANY")

    # 8. DEFAULT: no positive structural evidence found → REJECT
    #    This is the critical policy change: we do NOT accept unknown multi-word
    #    entities by default. Only T1/T2/T3 are accepted without blocklist hit.
    return ValidationResult(False, 0.0, False, "no_positive_evidence", "COMPANY")


# ---------------------------------------------------------------------------
# Detectors
# ---------------------------------------------------------------------------

def detect_person_names(block: TextBlock) -> list[PIIMatch]:
    """
    Detect full person names using spaCy NER (PERSON label).
    All candidates pass through _validate_person() before emission.
    Company-suffix conflict resolution redirects mislabelled entities.
    """
    nlp = load_nlp_model()
    doc = nlp(block.text)
    matches: list[PIIMatch] = []

    for ent in doc.ents:
        if ent.label_ != "PERSON":
            continue
        preceding = _get_preceding_context(doc, ent)
        result = _validate_person(ent.text, block.text, preceding)
        if result.accepted:
            matches.append(PIIMatch(
                text=_clean_entity_text(ent.text),
                category=result.suggested_category,
                start_char=ent.start_char,
                end_char=ent.end_char,
                source_block=block,
                confidence=result.confidence,
                needs_review=result.needs_review,
                detector="ner",
            ))
    return matches


def detect_organizations(block: TextBlock) -> list[PIIMatch]:
    """
    Detect company / organisation names using spaCy NER (ORG label).
    spaCy ORG maps to assignment category COMPANY.
    All candidates pass through _validate_company() before emission.
    Only entities with explicit T1/T2/T3 structural evidence are accepted.
    """
    nlp = load_nlp_model()
    doc = nlp(block.text)
    matches: list[PIIMatch] = []

    for ent in doc.ents:
        if ent.label_ != "ORG":
            continue
        preceding = _get_preceding_context(doc, ent)
        result = _validate_company(ent.text, block.text, preceding)
        if result.accepted:
            matches.append(PIIMatch(
                text=_clean_entity_text(ent.text),
                category=result.suggested_category,
                start_char=ent.start_char,
                end_char=ent.end_char,
                source_block=block,
                confidence=result.confidence,
                needs_review=result.needs_review,
                detector="ner",
            ))
    return matches


# ---------------------------------------------------------------------------
# ADDRESS detector  (heuristic / best-effort)
# ---------------------------------------------------------------------------

_ADDR_KEYWORD_RE = re.compile(
    r"\b(?:street|road|avenue|lane|plot|flat|floor|building|nagar|sector|"
    r"colony|taluka|district|village|pin|zip|postal|po\s+box|p\.o\.\s*box|"
    r"phase|block|wing|industrial\s+estate|office|suite|unit)\b",
    re.IGNORECASE,
)

_POSTAL_RE = re.compile(r"\b\d{6}\b|\b\d{5}(?:[-\s]\d{4})?\b")
_LOCATION_LABELS = {"GPE", "LOC", "FAC"}
_MIN_EVIDENCE = 2


def _count_address_evidence(text: str, spacy_ents) -> int:
    score = 0
    if any(e.label_ in _LOCATION_LABELS for e in spacy_ents):
        score += 1
    if _ADDR_KEYWORD_RE.search(text):
        score += 1
    if _POSTAL_RE.search(text):
        score += 1
    return score


def detect_addresses(block: TextBlock) -> list[PIIMatch]:
    """
    Detect physical/mailing addresses using spatial clustering of independent signals.
    Requires at least 2 distinct signal types (Keyword, Postal Code, Location Entity).
    Extracts a bounded contiguous span instead of returning the entire block.
    """
    nlp = load_nlp_model()
    doc = nlp(block.text)
    text = block.text

    signals = []
    # 1. Location entities from spaCy
    for ent in doc.ents:
        if ent.label_ in _LOCATION_LABELS:
            signals.append((ent.start_char, ent.end_char, "LOC"))
    
    # 2. Address Keywords
    for m in _ADDR_KEYWORD_RE.finditer(text):
        signals.append((m.start(), m.end(), "KW"))
        
    # 3. Postal Codes
    for m in _POSTAL_RE.finditer(text):
        signals.append((m.start(), m.end(), "PIN"))
        
    if not signals:
        return []
        
    # Sort signals by start position
    signals.sort(key=lambda x: x[0])
    
    # Cluster signals that are within 60 characters of each other
    clusters = []
    current_cluster = [signals[0]]
    
    for sig in signals[1:]:
        prev = current_cluster[-1]
        # Distance from end of previous signal to start of current signal
        if sig[0] - prev[1] <= 60:
            current_cluster.append(sig)
        else:
            clusters.append(current_cluster)
            current_cluster = [sig]
    clusters.append(current_cluster)
    
    matches = []
    for cluster in clusters:
        types = set(s[2] for s in cluster)
        if len(types) >= 2:
            min_start = min(s[0] for s in cluster)
            max_end = max(s[1] for s in cluster)
            
            # Expand backwards to start of string or nearest sentence boundary
            prefix = text[:min_start]
            match_start = re.search(r'[\.\;\n](?!.*[\.\;\n])', prefix)
            start_idx = match_start.end() if match_start else 0
                
            # Expand forwards to end of string or nearest sentence boundary
            suffix = text[max_end:]
            match_end = re.search(r'[\.\;\n]', suffix)
            end_idx = max_end + match_end.start() if match_end else len(text)
                
            # Strip whitespace
            while start_idx < end_idx and text[start_idx].isspace():
                start_idx += 1
            while end_idx > start_idx and text[end_idx-1].isspace():
                end_idx -= 1
                
            # Enforce max length of 300 characters to prevent matching whole paragraphs
            if end_idx - start_idx > 300:
                # If too long, just clamp it tightly around the signals
                start_idx = min_start
                end_idx = max_end
                
            if start_idx < end_idx:
                span_text = text[start_idx:end_idx]
                matches.append(PIIMatch(
                    text=span_text,
                    category="ADDRESS",
                    start_char=start_idx,
                    end_char=end_idx,
                    source_block=block,
                    confidence=0.85,
                    needs_review=True,
                    detector="heuristic",
                ))
                
    return matches


# ---------------------------------------------------------------------------
# Convenience runner
# ---------------------------------------------------------------------------

def run_all_ner_detectors(block: TextBlock) -> list[PIIMatch]:
    results: list[PIIMatch] = []
    results.extend(detect_person_names(block))
    results.extend(detect_organizations(block))
    results.extend(detect_addresses(block))
    return results
