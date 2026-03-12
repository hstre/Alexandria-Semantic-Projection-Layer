"""
Gateway result and audit types (WP2 §2, §7.2).

GatewayEvent  — single audit record per candidate emission/rejection
SPLResult     — result of a single-builder submission
DualBuilderResult — result of a dual-builder (E4) submission
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional

from spl import ClaimCandidate, EmissionRule, EmissionStatus


# ── Audit type ────────────────────────────────────────────────────────────────

@dataclass
class GatewayEvent:
    """
    A single audit event produced by emit_claim_nodes().

    Every candidate that passes through the gateway — whether emitted or
    rejected — produces one GatewayEvent. These are persisted to
    audit_log.json (JSON Lines format) for downstream auditability.

    Fields
    ------
    candidate_id    ID of the ClaimCandidate
    emission_rule   E1 | E2 | (rule of the candidate)
    thresholds      Snapshot of Θ active at emission time
    decision        "EMITTED" | "REJECTED"
    reason          Empty string if EMITTED; rejection reason if REJECTED
    claim_id        SHA256 claim hash if EMITTED; empty string if REJECTED
    timestamp       Unix timestamp
    """
    candidate_id:  str
    emission_rule: str
    thresholds:    dict
    decision:      str
    reason:        str
    claim_id:      str = ""
    timestamp:     float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        return {
            "candidate_id":  self.candidate_id,
            "emission_rule": self.emission_rule,
            "thresholds":    self.thresholds,
            "decision":      self.decision,
            "reason":        self.reason,
            "claim_id":      self.claim_id,
            "timestamp":     self.timestamp,
        }


# ── Result types ──────────────────────────────────────────────────────────────

@dataclass
class SPLResult:
    """
    The result of submitting one SemanticProjection to the SPLGateway.

    The protocol layer inspects .status to decide how to proceed:

        READY_FOR_CLAIM      → call gateway.emit_claim_nodes(result.candidates)
        AMBIGUOUS            → no claims; log for human review or re-projection
        STRUCTURAL_VIOLATION → blocked by ontological shield (E0)
        BRANCH_CANDIDATE     → E4; inspect DualBuilderResult

    Fields
    ------
    result_id         Unique ID for this result (for audit cross-reference)
    unit_id           Back-reference to originating SemanticUnit
    projection_id     Back-reference to originating SemanticProjection
    status            EmissionStatus after E0–E3 evaluation
    emission_rule     Which rule fired (None if not yet evaluated)
    candidates        List of ClaimCandidates (empty for E0/E3)
    h_norm            H_norm of the projection
    builder_origin    "alpha" | "beta"
    matrix_version    Relation matrix version used
    submitted_at      Timestamp of gateway submission
    """
    result_id:       str
    unit_id:         str
    projection_id:   str
    status:          EmissionStatus
    emission_rule:   Optional[EmissionRule]
    candidates:      list[ClaimCandidate]
    h_norm:          float
    builder_origin:  str
    matrix_version:  str
    submitted_at:    float = field(default_factory=time.time)

    def is_ready(self) -> bool:
        """True if candidates can be passed to emit_claim_nodes()."""
        return self.status == EmissionStatus.READY_FOR_CLAIM

    def is_blocked(self) -> bool:
        """True if blocked (E0 or E3). No claims will be emitted."""
        return self.status in (
            EmissionStatus.AMBIGUOUS,
            EmissionStatus.STRUCTURAL_VIOLATION,
        )

    def is_branched(self) -> bool:
        """True if E4 was applied (BRANCH_CANDIDATE)."""
        return self.status == EmissionStatus.BRANCH_CANDIDATE

    def top_candidate(self) -> Optional[ClaimCandidate]:
        """Return rank-1 candidate, or None if blocked."""
        for c in self.candidates:
            if c.rank == 1:
                return c
        return self.candidates[0] if self.candidates else None

    def to_dict(self) -> dict:
        return {
            "result_id":       self.result_id,
            "unit_id":         self.unit_id,
            "projection_id":   self.projection_id,
            "status":          self.status.value,
            "emission_rule":   self.emission_rule.value if self.emission_rule else None,
            "h_norm":          round(self.h_norm, 6),
            "candidate_count": len(self.candidates),
            "builder_origin":  self.builder_origin,
            "matrix_version":  self.matrix_version,
            "submitted_at":    self.submitted_at,
            "candidates":      [c.to_dict() for c in self.candidates],
        }


@dataclass
class DualBuilderResult:
    """
    The result of submitting the same SemanticUnit projected by two builders.

    Produced by SPLGateway.submit_dual(). E4 (JSD check) is applied first.

    Protocol guidance (WP2 §7.2):
        branched=False → pass result_alpha.candidates to emit_claim_nodes()
        branched=True  → the protocol must decide: branch or adjudicate.
                         Do NOT call emit_claim_nodes() on BRANCH_CANDIDATE results.
    """
    dual_id:       str
    unit_id:       str
    jsd:           float
    branched:      bool
    result_alpha:  SPLResult
    result_beta:   SPLResult
    submitted_at:  float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        return {
            "dual_id":      self.dual_id,
            "unit_id":      self.unit_id,
            "jsd":          round(self.jsd, 6),
            "branched":     self.branched,
            "result_alpha": self.result_alpha.to_dict(),
            "result_beta":  self.result_beta.to_dict(),
            "submitted_at": self.submitted_at,
        }
