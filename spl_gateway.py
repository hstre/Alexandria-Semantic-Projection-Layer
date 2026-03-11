"""
Alexandria — SPL Gateway
========================
spl_gateway.py

Working Paper 2: "Semantic Projection Layer — A Formal Bridge between
Natural Language and Epistemic Protocol" (Rentschler, v16)

This module is the SINGLE ENTRY POINT for the Alexandria Protocol to call
into the Semantic Projection Layer. The protocol layer imports only this
module — it never constructs EmissionEngine or ClaimCandidateConverter
directly.

Architecture (WP2 §2):

    [Protocol Layer]
         ↓  calls
    [SPLGateway]              ← this module
         ↓  uses
    [EmissionEngine]          ← spl.py
    [ClaimCandidateConverter] ← spl.py
         ↓  produces
    [ClaimNode]               ← schema.py (protocol-side)

Public API
----------
    SPLGateway          — main gateway class, instantiate once per session
    SPLResult           — result of a single-builder submission
    DualBuilderResult   — result of a dual-builder (E4) submission
    SPLGatewayError     — raised on protocol violations at the boundary

Protocol invariants enforced here [SHALL]:
    1. Only E1/E2 candidates reach the protocol (READY_FOR_CLAIM gate)
    2. E3 (AMBIGUOUS) is always blocked — protocol receives empty list
    3. E0 (STRUCTURAL_VIOLATION) is always blocked — protocol receives empty list
    4. E4 (BRANCH_CANDIDATE) triggers dual result — protocol decides branching
    5. Every ClaimNode carries SPL provenance in source_refs (WP2 §7.4)
    6. All submissions are logged in the audit trail (WP2 §7.4)

Reference: WP2 §2, §7, Appendix I
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Optional

from spl import (
    ClaimCandidate,
    ClaimCandidateConverter,
    EmissionEngine,
    EmissionRule,
    EmissionStatus,
    SemanticProjection,
    SemanticUnit,
    SPLThresholds,
    compute_jsd,
)


# ── Exceptions ────────────────────────────────────────────────────────────────

class SPLGatewayError(Exception):
    """
    Raised when the protocol layer violates a gateway invariant.

    Examples:
        - Submitting a non-E1/E2 candidate for conversion
        - Submitting projections from different SemanticUnits to submit_dual
        - Submitting a projection with invalid P_r (not summing to ~1.0)
    """


# ── Result Types ──────────────────────────────────────────────────────────────

@dataclass
class SPLResult:
    """
    The result of submitting one SemanticProjection to the SPLGateway.

    The protocol layer inspects .status to decide how to proceed:

        READY_FOR_CLAIM      → call gateway.to_claims(result) to get ClaimNodes
        AMBIGUOUS            → no claims; log for human review or re-projection
        STRUCTURAL_VIOLATION → blocked by ontological shield (E0)
        BRANCH_CANDIDATE     → E4 was applied externally; inspect DualBuilderResult

    Fields
    ------
    result_id         Unique ID for this result (for audit cross-reference)
    unit_id           Back-reference to originating SemanticUnit
    projection_id     Back-reference to originating SemanticProjection
    status            EmissionStatus after E0–E3 evaluation
    emission_rule     Which rule fired (None if projection was not yet evaluated)
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

    # ── Convenience predicates ─────────────────────────────────────────────

    def is_ready(self) -> bool:
        """True if candidates can be converted to ClaimNodes."""
        return self.status == EmissionStatus.READY_FOR_CLAIM

    def is_blocked(self) -> bool:
        """True if the projection was blocked (E0 or E3). No claims emitted."""
        return self.status in (
            EmissionStatus.AMBIGUOUS,
            EmissionStatus.STRUCTURAL_VIOLATION,
        )

    def is_branched(self) -> bool:
        """True if E4 was applied (BRANCH_CANDIDATE). Use DualBuilderResult."""
        return self.status == EmissionStatus.BRANCH_CANDIDATE

    def top_candidate(self) -> Optional[ClaimCandidate]:
        """Return rank-1 candidate, or None if blocked."""
        for c in self.candidates:
            if c.rank == 1:
                return c
        return self.candidates[0] if self.candidates else None

    def to_dict(self) -> dict:
        return {
            "result_id":      self.result_id,
            "unit_id":        self.unit_id,
            "projection_id":  self.projection_id,
            "status":         self.status.value,
            "emission_rule":  self.emission_rule.value if self.emission_rule else None,
            "h_norm":         round(self.h_norm, 6),
            "candidate_count": len(self.candidates),
            "builder_origin": self.builder_origin,
            "matrix_version": self.matrix_version,
            "submitted_at":   self.submitted_at,
            "candidates":     [c.to_dict() for c in self.candidates],
        }


@dataclass
class DualBuilderResult:
    """
    The result of submitting the same SemanticUnit projected by two builders.

    Produced by SPLGateway.submit_dual(). The gateway applies E4 (JSD check)
    before returning individual SPLResults for alpha and beta.

    Fields
    ------
    dual_id        Unique ID for this dual result
    unit_id        The shared SemanticUnit.unit_id
    jsd            Jensen-Shannon Divergence between P_rᴬ and P_rᴮ (WP2 §3.3.5)
    branched       True if JSD > τ₄ → both projections are BRANCH_CANDIDATE
    result_alpha   SPLResult for builder "alpha"
    result_beta    SPLResult for builder "beta"
    submitted_at   Timestamp

    Protocol guidance (WP2 §7.2):
        branched=False → use result_alpha (or merge, per governance rules)
        branched=True  → the protocol must decide: branch or adjudicate
                         Do NOT call to_claims() on BRANCH_CANDIDATE results.
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


# ── SPLGateway ────────────────────────────────────────────────────────────────

class SPLGateway:
    """
    The formal interface between the Alexandria Protocol and the SPL.

    This is the SINGLE entry point for the protocol layer. The protocol
    never constructs EmissionEngine, ClaimCandidateConverter, or emission
    rules directly.

    The gateway enforces all protocol invariants defined in WP2 §7:
        - Only READY_FOR_CLAIM results produce ClaimNodes
        - BRANCH_CANDIDATE results are returned as DualBuilderResult
        - All submissions are recorded in the audit log

    Instantiation
    -------------
        gateway = SPLGateway()                        # default Θ
        gateway = SPLGateway(thresholds=custom_theta) # custom calibration

    Single-builder workflow
    -----------------------
        result = gateway.submit(projection)
        if result.is_ready():
            claims = gateway.to_claims(result)

    Dual-builder workflow (E4)
    --------------------------
        dual = gateway.submit_dual(proj_alpha, proj_beta)
        if not dual.branched:
            claims = gateway.to_claims(dual.result_alpha)
        else:
            # Protocol decides: branch or adjudicate
            handle_branch(dual)

    Batch workflow
    --------------
        results = gateway.submit_batch(projections)
        ready   = [r for r in results if r.is_ready()]
        claims  = gateway.to_claims_batch(ready)

    Audit
    -----
        log = gateway.audit_log()   # list of dicts, one per submission
    """

    def __init__(self, thresholds: SPLThresholds | None = None):
        self._theta     = thresholds or SPLThresholds()
        self._engine    = EmissionEngine(self._theta)
        self._converter = ClaimCandidateConverter()
        self._log:  list[dict] = []

        errors = self._theta.validate()
        if errors:
            raise SPLGatewayError(
                f"Invalid SPLThresholds: {'; '.join(errors)}"
            )

    # ── Properties ────────────────────────────────────────────────────────────

    @property
    def thresholds(self) -> SPLThresholds:
        return self._theta

    # ── Core submission methods ───────────────────────────────────────────────

    def submit(
        self,
        projection: SemanticProjection,
        k: int = 3,
    ) -> SPLResult:
        """
        Submit a single SemanticProjection to the emission engine.

        Applies E0 → E3 in order (per WP2 §7.2). Updates projection.status
        and projection.emission_rule in place. Returns SPLResult.

        Parameters
        ----------
        projection  A SemanticProjection produced by the NLP backend.
        k           Max candidates for E2 (default 3, per WP2 Appendix I.1).

        Returns
        -------
        SPLResult with status, candidates, and audit metadata.
        """
        self._validate_projection(projection)

        candidates = self._engine.emit(projection, k=k)

        result = SPLResult(
            result_id=str(uuid.uuid4()),
            unit_id=projection.unit_id,
            projection_id=projection.projection_id,
            status=projection.status,
            emission_rule=projection.emission_rule,
            candidates=candidates,
            h_norm=projection.h_norm,
            builder_origin=projection.builder_origin,
            matrix_version=projection.matrix_version,
        )

        self._record(result)
        return result

    def submit_dual(
        self,
        proj_alpha: SemanticProjection,
        proj_beta:  SemanticProjection,
        k: int = 3,
    ) -> DualBuilderResult:
        """
        Submit projections from two independent builders for the same SemanticUnit.

        Applies E4 (JSD check) first. If JSD > τ₄, both projections become
        BRANCH_CANDIDATE and the protocol must decide. Otherwise, E0–E3 are
        applied to each projection independently.

        Parameters
        ----------
        proj_alpha  Projection from builder "alpha"
        proj_beta   Projection from builder "beta"
        k           Max candidates for E2

        Returns
        -------
        DualBuilderResult with jsd, branched flag, and individual SPLResults.

        Raises
        ------
        SPLGatewayError if unit_id does not match between the two projections.
        """
        if proj_alpha.unit_id != proj_beta.unit_id:
            raise SPLGatewayError(
                "submit_dual requires both projections to share the same unit_id. "
                f"Got alpha.unit_id={proj_alpha.unit_id!r}, "
                f"beta.unit_id={proj_beta.unit_id!r}. "
                "Each SemanticUnit must be submitted as a pair."
            )

        self._validate_projection(proj_alpha)
        self._validate_projection(proj_beta)

        # E4: JSD check — applied BEFORE individual E0–E3 evaluation
        jsd = self._engine.apply_e4(proj_alpha, proj_beta)
        branched = jsd > self._theta.tau_4

        if branched:
            # E4 fired: both projections are BRANCH_CANDIDATE.
            # Do NOT call submit() — it would re-run emit() and overwrite E4 status.
            result_alpha = self._make_branch_result(proj_alpha)
            result_beta  = self._make_branch_result(proj_beta)
            self._record(result_alpha)
            self._record(result_beta)
        else:
            # JSD within tolerance — evaluate E0–E3 independently per builder
            result_alpha = self.submit(proj_alpha, k=k)
            result_beta  = self.submit(proj_beta,  k=k)

        dual = DualBuilderResult(
            dual_id=str(uuid.uuid4()),
            unit_id=proj_alpha.unit_id,
            jsd=jsd,
            branched=branched,
            result_alpha=result_alpha,
            result_beta=result_beta,
        )

        self._log.append({
            "event":       "submit_dual",
            "dual_id":     dual.dual_id,
            "unit_id":     dual.unit_id,
            "jsd":         round(jsd, 6),
            "branched":    branched,
            "tau_4":       self._theta.tau_4,
            "timestamp":   dual.submitted_at,
        })

        return dual

    def submit_batch(
        self,
        projections: list[SemanticProjection],
        k: int = 3,
    ) -> list[SPLResult]:
        """
        Submit multiple projections. Each is evaluated independently.
        Order is preserved. Useful for processing all units of a document.
        """
        return [self.submit(p, k=k) for p in projections]

    # ── Protocol boundary: SPLResult → ClaimNode ──────────────────────────────

    def to_claims(
        self,
        result: SPLResult,
        extra_assumptions: list[str] | None = None,
    ) -> list:
        """
        Convert a READY_FOR_CLAIM SPLResult to ClaimNodes (protocol boundary).

        This is the only legal path from SPLResult to ClaimNode.
        Internally calls ClaimCandidateConverter (WP2 §2).

        Parameters
        ----------
        result             An SPLResult with status READY_FOR_CLAIM.
        extra_assumptions  Additional assumptions to attach to each ClaimNode.

        Returns
        -------
        List of ClaimNodes (may be empty if no E1/E2 candidates).

        Raises
        ------
        SPLGatewayError if result.status is not READY_FOR_CLAIM.
        """
        if not result.is_ready():
            raise SPLGatewayError(
                f"Cannot convert result with status={result.status.value} to ClaimNodes. "
                f"Only READY_FOR_CLAIM results are convertible. "
                f"(result_id={result.result_id}, unit_id={result.unit_id})"
            )

        claims = self._converter.convert_batch(
            result.candidates, extra_assumptions=extra_assumptions
        )

        self._log.append({
            "event":       "to_claims",
            "result_id":   result.result_id,
            "unit_id":     result.unit_id,
            "claim_count": len(claims),
            "timestamp":   time.time(),
        })

        return claims

    def to_claims_batch(
        self,
        results: list[SPLResult],
        extra_assumptions: list[str] | None = None,
    ) -> list:
        """
        Convert a list of READY_FOR_CLAIM results to ClaimNodes.
        Skips blocked/branched results silently (logs each skip).
        """
        all_claims = []
        for result in results:
            if result.is_ready():
                all_claims.extend(
                    self.to_claims(result, extra_assumptions=extra_assumptions)
                )
            else:
                self._log.append({
                    "event":     "to_claims_batch_skip",
                    "result_id": result.result_id,
                    "status":    result.status.value,
                    "timestamp": time.time(),
                })
        return all_claims

    # ── Audit ─────────────────────────────────────────────────────────────────

    def audit_log(self) -> list[dict]:
        """
        Return the complete audit log for this gateway session.

        Each entry is a dict with at minimum:
            event, timestamp

        Event types:
            "submit"        — single-builder submission
            "submit_dual"   — dual-builder submission (E4)
            "to_claims"     — conversion to ClaimNodes
            "to_claims_batch_skip" — skipped non-ready result

        WP2 §7.4: The audit log is the protocol's guarantee that no claim
        entered the system without passing through the SPL emission rules.
        """
        return list(self._log)

    def summary(self) -> dict:
        """
        Return a summary of all submissions in this gateway session.

        Counts per EmissionStatus and per EmissionRule, plus conversion stats.
        """
        statuses: dict[str, int] = {}
        rules:    dict[str, int] = {}
        total_candidates = 0
        total_claims = 0

        for entry in self._log:
            if entry["event"] == "submit":
                s = entry.get("status", "unknown")
                statuses[s] = statuses.get(s, 0) + 1
                r = entry.get("emission_rule")
                if r:
                    rules[r] = rules.get(r, 0) + 1
                total_candidates += entry.get("candidate_count", 0)
            elif entry["event"] == "to_claims":
                total_claims += entry.get("claim_count", 0)

        return {
            "submissions":       statuses.get("ready_for_claim", 0)
                                 + statuses.get("ambiguous", 0)
                                 + statuses.get("structural_violation", 0)
                                 + statuses.get("branch_candidate", 0),
            "by_status":         statuses,
            "by_emission_rule":  rules,
            "total_candidates":  total_candidates,
            "total_claims":      total_claims,
            "thresholds":        {
                "tau_0": self._theta.tau_0,
                "tau_1": self._theta.tau_1,
                "tau_2": self._theta.tau_2,
                "tau_3": self._theta.tau_3,
                "tau_4": self._theta.tau_4,
            },
        }

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _make_branch_result(self, projection: SemanticProjection) -> SPLResult:
        """Create a BRANCH_CANDIDATE SPLResult without re-running emit()."""
        return SPLResult(
            result_id=str(uuid.uuid4()),
            unit_id=projection.unit_id,
            projection_id=projection.projection_id,
            status=EmissionStatus.BRANCH_CANDIDATE,
            emission_rule=EmissionRule.E4,
            candidates=[],
            h_norm=projection.h_norm,
            builder_origin=projection.builder_origin,
            matrix_version=projection.matrix_version,
        )

    def _validate_projection(self, projection: SemanticProjection) -> None:
        """Basic structural validation before emission."""
        if not projection.P_r:
            raise SPLGatewayError(
                f"projection.P_r is empty for projection_id={projection.projection_id}. "
                "An NLP backend must produce a non-empty relational distribution."
            )
        total = sum(projection.P_r.values())
        if abs(total - 1.0) > 0.01:
            raise SPLGatewayError(
                f"projection.P_r does not sum to 1.0 (got {total:.4f}) "
                f"for projection_id={projection.projection_id}."
            )

    def _record(self, result: SPLResult) -> None:
        """Append a submit event to the audit log."""
        self._log.append({
            "event":          "submit",
            "result_id":      result.result_id,
            "unit_id":        result.unit_id,
            "projection_id":  result.projection_id,
            "status":         result.status.value,
            "emission_rule":  result.emission_rule.value if result.emission_rule else None,
            "h_norm":         round(result.h_norm, 6),
            "candidate_count": len(result.candidates),
            "builder_origin": result.builder_origin,
            "matrix_version": result.matrix_version,
            "timestamp":      result.submitted_at,
        })


# ── Convenience factory ───────────────────────────────────────────────────────

def make_gateway(
    tau_0: float = 0.50,
    tau_1: float = 0.60,
    tau_2: float = 0.25,
    tau_3: float = 0.65,
    tau_4: float = 0.40,
) -> SPLGateway:
    """
    Factory function for constructing a calibrated SPLGateway.

    Useful for domain-specific calibration without importing SPLThresholds:

        gateway = make_gateway(tau_1=0.70, tau_2=0.20)  # stricter E1

    Returns an SPLGateway with the given threshold parameters.
    Raises SPLGatewayError if the thresholds are invalid.
    """
    theta = SPLThresholds(
        tau_0=tau_0, tau_1=tau_1,
        tau_2=tau_2, tau_3=tau_3,
        tau_4=tau_4,
    )
    return SPLGateway(thresholds=theta)
