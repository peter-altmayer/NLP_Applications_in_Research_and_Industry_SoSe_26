"""Lightweight regex-based PHI masker.

Our datasets (BioASQ, PubMedQA, SQuAD) contain no real patient data,
so this module is illustrative of what a production pipeline would need.
It masks common PHI patterns with a category tag.
"""
import re
from typing import Tuple

# (pattern, replacement_tag)
_RULES: list[Tuple[re.Pattern, str]] = [
    (re.compile(r"\b\d{3}-\d{2}-\d{4}\b"), "[SSN]"),
    (re.compile(r"\b(?:\+?1[\s.-]?)?\(?\d{3}\)?[\s.-]?\d{3}[\s.-]?\d{4}\b"), "[PHONE]"),
    (re.compile(r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b"), "[EMAIL]"),
    # Dates: MM/DD/YYYY, DD-MM-YYYY, Month DD YYYY, etc.
    (re.compile(
        r"\b(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|"
        r"Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)"
        r"\.?\s+\d{1,2},?\s+\d{4}\b",
        re.IGNORECASE,
    ), "[DATE]"),
    (re.compile(r"\b\d{1,2}[/\-]\d{1,2}[/\-]\d{2,4}\b"), "[DATE]"),
    # Patient / subject IDs
    (re.compile(r"\b(?:patient|subject|pt|case)\s*(?:id|#|no\.?)?\s*[:=]?\s*\d+\b", re.IGNORECASE), "[PATIENT_ID]"),
    # MRN (medical record number)
    (re.compile(r"\bMRN\s*[:=]?\s*\d+\b", re.IGNORECASE), "[MRN]"),
    # Ages (retain for clinical utility — commented out; uncomment if stricter)
    # (re.compile(r"\b\d{1,3}[\s-]?(?:year|yr)s?[\s-]?old\b", re.IGNORECASE), "[AGE]"),
]


def mask_phi(text: str) -> str:
    """Replace PHI patterns in *text* with category tags."""
    for pattern, tag in _RULES:
        text = pattern.sub(tag, text)
    return text


def mask_phi_batch(texts: list[str]) -> list[str]:
    return [mask_phi(t) for t in texts]


def phi_detected(text: str) -> bool:
    """Return True if any PHI pattern is present in *text*."""
    return any(p.search(text) for p, _ in _RULES)
