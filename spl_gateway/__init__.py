"""
Alexandria — SPL Gateway
========================
spl_gateway

Working Paper 2: "Semantic Projection Layer — A Formal Bridge between
Natural Language and Epistemic Protocol" (Rentschler, v16)

This package is the SINGLE ENTRY POINT for the Alexandria Protocol to call
into the Semantic Projection Layer. The protocol layer imports only this
package — it never constructs EmissionEngine or ClaimCandidateConverter
directly.

Architecture (WP2 §2):

    [Protocol Layer]
         ↓  calls
    [SPLGateway]              ← this package (legal protocol entry point)
         ↓  uses
    [EmissionEngine]          ← spl.py        (probabilistic pre-protocol stage)
    [ClaimCandidateConverter] ← _converter.py (boundary conversion, this package)
         ↓  produces
    [ClaimNode]               ← schema.py (protocol-side)

Package layout
--------------
    _exceptions.py   SPLGatewayError, CandidateRejectedError, ClaimValidationError
    _types.py        GatewayEvent, SPLResult, DualBuilderResult
    _utils.py        canonicalize_text, canonicalize_entities, hash_claim,
                     validate_claim_node
    _converter.py    _CATEGORY_HINT_MAP, _MODALITY_HINT_MAP,
                     validate_candidate_for_protocol_entry, ClaimCandidateConverter
    _gateway.py      MIN_EVIDENCE, SPLGateway, make_gateway

Public API
----------
    SPLGateway              — main gateway class, instantiate once per session
    SPLResult               — result of a single-builder submission
    DualBuilderResult       — result of a dual-builder (E4) submission
    GatewayEvent            — single audit event record
    SPLGatewayError         — raised on protocol violations at the boundary
    CandidateRejectedError  — raised when a candidate fails gateway validation
    ClaimValidationError    — raised when a ClaimNode fails structural validation
    canonicalize_text()                      — text normalisation
    canonicalize_entities()                  — entity normalisation
    hash_claim()                             — deterministic SHA256 claim identity
    validate_claim_node()                    — structural ClaimNode validator
    validate_candidate_for_protocol_entry()  — explicit pre-conversion boundary check
    ClaimCandidateConverter                  — the ONLY legal ClaimCandidate→ClaimNode path

Protocol invariants enforced here [SHALL]:
    1. emit_claim_nodes() is the only legal path from ClaimCandidate to ClaimNode
    2. Every candidate is validated before conversion
    3. Every ClaimNode is structurally validated after conversion
    4. Every ClaimNode carries a deterministic claim_id (SHA256)
    5. Every emission event is persisted to audit_log.json
    6. E3 / E0 / E4 results are hard-blocked at the gateway

Reference: WP2 §2, §7, Appendix I
"""

from spl_gateway._exceptions import (
    SPLGatewayError,
    CandidateRejectedError,
    ClaimValidationError,
)
from spl_gateway._types import (
    GatewayEvent,
    SPLResult,
    DualBuilderResult,
)
from spl_gateway._utils import (
    canonicalize_text,
    canonicalize_entities,
    hash_claim,
    validate_claim_node,
)
from spl_gateway._converter import (
    _CATEGORY_HINT_MAP,
    _MODALITY_HINT_MAP,
    validate_candidate_for_protocol_entry,
    ClaimCandidateConverter,
)
from spl_gateway._gateway import (
    MIN_EVIDENCE,
    SPLGateway,
    make_gateway,
)

__all__ = [
    # Exceptions
    "SPLGatewayError",
    "CandidateRejectedError",
    "ClaimValidationError",
    # Audit / result types
    "GatewayEvent",
    "SPLResult",
    "DualBuilderResult",
    # Utilities
    "canonicalize_text",
    "canonicalize_entities",
    "hash_claim",
    "validate_claim_node",
    # Boundary
    "_CATEGORY_HINT_MAP",
    "_MODALITY_HINT_MAP",
    "validate_candidate_for_protocol_entry",
    "ClaimCandidateConverter",
    # Gateway
    "MIN_EVIDENCE",
    "SPLGateway",
    "make_gateway",
]
