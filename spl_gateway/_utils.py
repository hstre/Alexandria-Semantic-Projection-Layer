"""
Standalone utility functions for the gateway boundary.

canonicalize_text()     — text normalisation (lowercase, whitespace)
canonicalize_entities() — entity normalisation (removes parenthetical annotations)
hash_claim()            — deterministic SHA256 claim identity
validate_claim_node()   — structural ClaimNode validator
"""

from __future__ import annotations

import hashlib
import re

from spl_gateway._exceptions import ClaimValidationError


def canonicalize_text(text: str) -> str:
    """
    Normalize a text string: lowercase, strip leading/trailing whitespace,
    collapse internal whitespace to single spaces.

    Used by hash_claim() and canonicalize_entities() to ensure deterministic
    claim identity across equivalent surface forms.
    """
    return re.sub(r"\s+", " ", text.strip().lower())


def canonicalize_entities(entity: str) -> str:
    """
    Normalize an entity string for deterministic comparison.

    Applies canonicalize_text() plus removes parenthetical annotations
    (e.g., "Paris (city)" → "paris").
    """
    text = re.sub(r"\s*\(.*?\)", "", entity)
    return canonicalize_text(text)


def hash_claim(subject: str, predicate: str, obj: str) -> str:
    """
    Compute a deterministic SHA256 claim identity hash.

    Guarantees: identical (subject, predicate, object) → identical claim_id,
    regardless of surface-form variation that canonicalize_entities() collapses.

    Format: hex digest of SHA256(canonical_subject + "|" + canonical_predicate
                                  + "|" + canonical_object)

    This is the deterministic conversion guarantee required by the protocol:
    the same epistemic content always produces the same claim_id.
    """
    canonical = (
        canonicalize_entities(subject)
        + "|"
        + canonicalize_text(predicate)
        + "|"
        + canonicalize_entities(obj)
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def validate_claim_node(node) -> None:
    """
    Validate a ClaimNode for structural completeness.

    Raises ClaimValidationError if any required field is missing or empty.

    Required fields:
        subject    — must be a non-empty string
        predicate  — must be a non-empty string
        object     — must be a non-empty string
        source_refs — must be a non-empty list (SPL provenance)

    Invalid nodes must NOT enter the ClaimGraph (protocol invariant).
    """
    missing = []
    for attr in ("subject", "predicate", "object"):
        val = getattr(node, attr, None)
        if not val or not str(val).strip():
            missing.append(attr)

    source_refs = getattr(node, "source_refs", None)
    if not source_refs:
        missing.append("source_refs")

    if missing:
        raise ClaimValidationError(
            f"ClaimNode missing required fields: {missing}. "
            "Invalid nodes must not enter the ClaimGraph."
        )
