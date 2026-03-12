"""
Gateway exception hierarchy (WP2 §2).

All exceptions raised by spl_gateway are subclasses of SPLGatewayError.
"""


class SPLGatewayError(Exception):
    """Base exception for all gateway-level violations."""


class CandidateRejectedError(SPLGatewayError):
    """
    Raised when a ClaimCandidate fails the emit_claim_nodes() validation gate.

    Reasons include: non-EMIT rule, confidence < τ₁ (E1), entropy ≥ τ₂ (E1),
    entropy ≥ τ₃ (E2), JSD > τ₄, or evidence_count < MIN_EVIDENCE.
    """


class ClaimValidationError(SPLGatewayError):
    """
    Raised when a ClaimNode fails validate_claim_node().

    Reasons include: missing subject, predicate, object, or source_refs.
    Invalid nodes must not enter the ClaimGraph.
    """
