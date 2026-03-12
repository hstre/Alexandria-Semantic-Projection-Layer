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
    [SPLGateway]              ← this module  (legal protocol entry point)
         ↓  uses
    [EmissionEngine]          ← spl.py       (probabilistic pre-protocol stage)
    [ClaimCandidateConverter] ← spl_gateway.py (boundary conversion, this file)
         ↓  produces
    [ClaimNode]               ← schema.py (protocol-side)

Public API
----------
    SPLGateway              — main gateway class, instantiate once per session
    SPLResult               — result of a single-builder submission
    DualBuilderResult       — result of a dual-builder (E4) submission
    GatewayEvent            — single audit event record
    SPLGatewayError         — raised on protocol violations at the boundary
    CandidateRejectedError  — raised when a candidate fails gateway validation
    ClaimValidationError    — raised when a ClaimNode fails structural validation
    canonicalize_text()                    — text normalisation (lowercase, whitespace)
    canonicalize_entities()               — entity normalisation
    hash_claim()                          — deterministic SHA256 claim identity
    validate_claim_node()                 — structural ClaimNode validator
    validate_candidate_for_protocol_entry() — explicit pre-conversion boundary check
    ClaimCandidateConverter               — the ONLY legal ClaimCandidate→ClaimNode path

Protocol invariants enforced here [SHALL]:
    1. emit_claim_nodes() is the only legal path from ClaimCandidate to ClaimNode
    2. Every candidate is validated (emission rule, confidence, entropy, JSD,
       evidence count) before conversion
    3. Every ClaimNode is structurally validated after conversion
    4. Every ClaimNode carries a deterministic claim_id (SHA256)
    5. Every emission event is persisted to audit_log.json
    6. E3 / E0 / E4 results are hard-blocked at the gateway

Reference: WP2 §2, §7, Appendix I
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import time
import uuid
from dataclasses import dataclass, field
from typing import Optional

from spl import (
    ClaimCandidate,
    EmissionEngine,
    EmissionRule,
    EmissionStatus,
    SemanticProjection,
    SPLThresholds,
    compute_jsd,
)


# ── Constants ─────────────────────────────────────────────────────────────────

MIN_EVIDENCE: int = 1
"""Minimum evidence count for a candidate to pass gateway validation."""


# ── Protocol Boundary: Category and Modality maps ─────────────────────────────
#
# These maps live here (not in spl.py) because they encode Alexandria protocol
# semantics. The SPL provides *hints* — this gateway layer is responsible for
# translating them into canonical protocol types.

_CATEGORY_HINT_MAP: dict[str, str] = {
    "dynamic":     "EMPIRICAL",
    "statistical": "EMPIRICAL",
    "epistemic":   "EMPIRICAL",
    "model":       "MODEL",
    "normative":   "NORMATIVE",
    "ontic":       "EMPIRICAL",   # default for ontological relations
}
"""Maps SPL semantic_category_hint strings to Alexandria Category enum names."""

_MODALITY_HINT_MAP: dict[str, str] = {
    "asserted":     "established",
    "suggested":    "evidence",
    "hypothesized": "hypothesis",
    "possible":     "suggestion",
}
"""Maps SPL modality_hint strings to Alexandria Modality enum values."""


# ── Exceptions ────────────────────────────────────────────────────────────────

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


# ── Audit Types ───────────────────────────────────────────────────────────────

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


# ── Result Types ──────────────────────────────────────────────────────────────

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


# ── Standalone utility functions ──────────────────────────────────────────────

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


# ── Protocol Boundary: explicit pre-conversion validation ────────────────────

def validate_candidate_for_protocol_entry(
    candidate: ClaimCandidate,
    thresholds: SPLThresholds | None = None,
) -> None:
    """
    Explicit boundary check before a ClaimCandidate may be converted to ClaimNode.

    This function makes the protocol invariant visible at the call site.
    Call this before ClaimCandidateConverter.convert() when you want an explicit
    gate rather than relying on the converter's internal check.

    Raises CandidateRejectedError if any criterion is not met:
        1. emission_rule ∈ {E1, E2}           — only EMIT candidates are convertible
        2. relation_score ≥ τ₁  (E1 only)     — confidence floor
        3. h_norm < τ₂          (E1 only)      — entropy ceiling for singular emission
        4. h_norm < τ₃          (E2 only)      — entropy ceiling for multiple emission

    Parameters
    ----------
    candidate    The ClaimCandidate to validate.
    thresholds   SPLThresholds Θ. Defaults to WP2 recommended values.
    """
    Θ = thresholds or SPLThresholds()

    if candidate.emission_rule not in (EmissionRule.E1, EmissionRule.E2):
        raise CandidateRejectedError(
            f"emission_rule={candidate.emission_rule.value} is not EMIT. "
            "Only E1/E2 candidates may cross the protocol boundary."
        )

    if candidate.emission_rule == EmissionRule.E1:
        if candidate.relation_score < Θ.tau_1:
            raise CandidateRejectedError(
                f"E1 confidence={candidate.relation_score:.4f} < τ₁={Θ.tau_1}"
            )
        if candidate.h_norm >= Θ.tau_2:
            raise CandidateRejectedError(
                f"E1 entropy h_norm={candidate.h_norm:.4f} ≥ τ₂={Θ.tau_2}"
            )
    else:  # E2
        if candidate.h_norm >= Θ.tau_3:
            raise CandidateRejectedError(
                f"E2 entropy h_norm={candidate.h_norm:.4f} ≥ τ₃={Θ.tau_3}"
            )


# ── Protocol Boundary: ClaimCandidateConverter ───────────────────────────────

class ClaimCandidateConverter:
    """
    The ONLY legal path from ClaimCandidate to ClaimNode.

    This class enforces the protocol boundary defined in WP2 §2:
    "The epistemic validity of all claims remains governed exclusively
    by the Alexandria protocol."

    Lives in spl_gateway.py (not spl.py) because it encodes protocol semantics:
    Category, Modality, and BuilderOrigin are Alexandria types, not SPL types.

    Rules enforced at the boundary [SHALL]:
    1. semantic_category_hint is converted to Alexandria Category.
       The SPL does not set Category — it provides hints.
    2. modality_hint is converted to Alexandria Modality.
    3. spl_provenance string is attached to source_refs for auditability.
    4. assumptions[] is always set (protocol invariant).
    5. The ClaimNode is NOT yet in VALIDATED status — it goes through
       the normal protocol cycle (PatchEmitter → AuditGate).

    [DBA] The converter is a DBA extension. A production system could
    replace it with a more sophisticated mapping (e.g. domain-specific
    category inference). The invariant is: ClaimCandidate must pass
    through SOME converter before becoming a ClaimNode.
    """

    def convert(
        self,
        candidate: ClaimCandidate,
        extra_assumptions: list[str] | None = None,
    ) -> "ClaimNode":
        """
        Convert a ClaimCandidate to a ClaimNode.

        Raises ValueError if candidate.emission_rule not in {E1, E2}.
        BRANCH_CANDIDATE and AMBIGUOUS must never be converted directly.
        """
        from .schema import ClaimNode, BuilderOrigin

        if candidate.emission_rule not in (EmissionRule.E1, EmissionRule.E2):
            raise ValueError(
                f"ClaimCandidate with emission_rule={candidate.emission_rule.value} "
                "cannot be converted to ClaimNode. "
                "Only E1/E2 (READY_FOR_CLAIM) candidates are convertible."
            )

        category    = self._map_category(candidate.semantic_category_hint)
        modality    = self._map_modality(candidate.modality_hint)
        origin      = BuilderOrigin.ALPHA if candidate.builder_origin == "alpha" else BuilderOrigin.BETA
        source_refs = [self._build_provenance(candidate)]
        assumptions = self._build_assumptions(candidate, extra_assumptions)

        claim = ClaimNode.new(
            subject=candidate.subject,
            predicate=candidate.relation,
            object=candidate.object,
            category=category,
            assumptions=assumptions,
            source_refs=source_refs,
        )
        claim.modality       = modality
        claim.builder_origin = origin
        return claim

    def _map_category(self, hint: str) -> "Category":
        """Map a semantic category hint to an Alexandria Category enum value."""
        from .schema import Category
        cat_value = _CATEGORY_HINT_MAP.get(hint.lower(), "EMPIRICAL")
        try:
            return Category[cat_value]
        except KeyError:
            return Category.EMPIRICAL

    def _map_modality(self, hint: str) -> "Modality":
        """Map a modality hint to an Alexandria Modality enum value."""
        from .schema import Modality
        mod_value = _MODALITY_HINT_MAP.get(hint.lower(), "hypothesis")
        try:
            return Modality(mod_value)
        except ValueError:
            return Modality.HYPOTHESIS

    def _build_provenance(self, candidate: ClaimCandidate) -> str:
        """Build the SPL provenance string for source_refs (WP2 §7.4)."""
        return (
            f"spl:{candidate.unit_id[:8]}/"
            f"{candidate.projection_id[:8]}/"
            f"E{candidate.emission_rule.value}/"
            f"score={candidate.relation_score:.3f}/"
            f"h={candidate.h_norm:.3f}/"
            f"matrix={candidate.matrix_version}"
        )

    def _build_assumptions(
        self,
        candidate: ClaimCandidate,
        extra: list[str] | None,
    ) -> list[str]:
        """Build the assumptions list [SHALL be non-empty]."""
        assumptions = list(extra or [])
        assumptions.append(
            f"SPL projection: relation={candidate.relation} "
            f"score={candidate.relation_score:.3f} "
            f"h_norm={candidate.h_norm:.3f}"
        )
        if candidate.scope_hint:
            assumptions.append(f"scope={candidate.scope_hint}")
        return assumptions

    def convert_batch(
        self,
        candidates: list[ClaimCandidate],
        extra_assumptions: list[str] | None = None,
    ) -> list["ClaimNode"]:
        """Convert a list of E1/E2 candidates. Skips non-convertible."""
        claims = []
        for c in candidates:
            if c.emission_rule in (EmissionRule.E1, EmissionRule.E2):
                try:
                    claims.append(self.convert(c, extra_assumptions))
                except (ValueError, KeyError, AttributeError):
                    pass
        return claims


# ── SPLGateway ────────────────────────────────────────────────────────────────

class SPLGateway:
    """
    The formal interface between the Alexandria Protocol and the SPL.

    This is the SINGLE entry point for the protocol layer.

    Key design principle:
        SPL is PROBABILISTIC  — it computes distributions over relation space
        Protocol is DETERMINISTIC — it operates on discrete, sealed ClaimNodes
        Gateway is the BOUNDARY  — it translates between the two regimes

    The boundary is enforced by emit_claim_nodes(), which is the ONLY legal
    path from ClaimCandidate to ClaimNode.

    Instantiation
    -------------
        gateway = SPLGateway()
        gateway = SPLGateway(thresholds=custom_theta, audit_log_path="my_log.json")

    Canonical workflow
    ------------------
        # 1. Submit projection (probabilistic)
        result = gateway.submit(projection)

        # 2. Emit to protocol (deterministic boundary)
        if result.is_ready():
            claims = gateway.emit_claim_nodes(result.candidates)

        # 3. Use claims in protocol
        for claim in claims:
            protocol.ingest(claim)

    Dual-builder workflow
    ---------------------
        dual = gateway.submit_dual(proj_alpha, proj_beta)
        if not dual.branched:
            claims = gateway.emit_claim_nodes(dual.result_alpha.candidates)
        else:
            handle_branch(dual)  # protocol decides

    Batch workflow
    --------------
        results = gateway.submit_batch(projections)
        claims  = gateway.emit_claims_from_results(results)

    Audit
    -----
        log = gateway.audit_log()          # in-memory events
        # audit_log.json                   # persisted GatewayEvents
    """

    def __init__(
        self,
        thresholds: SPLThresholds | None = None,
        audit_log_path: str | None = "audit_log.json",
    ):
        """
        Parameters
        ----------
        thresholds       SPLThresholds Θ. Defaults to WP2 recommended values.
        audit_log_path   Path for persisted GatewayEvent JSON Lines log.
                         Set to None to disable file persistence (in-memory only).
        """
        self._theta     = thresholds or SPLThresholds()
        self._engine    = EmissionEngine(self._theta)
        self._converter = ClaimCandidateConverter()
        self._log:  list[dict] = []
        self._audit_log_path = audit_log_path

        errors = self._theta.validate()
        if errors:
            raise SPLGatewayError(
                f"Invalid SPLThresholds: {'; '.join(errors)}"
            )

    # ── Properties ────────────────────────────────────────────────────────────

    @property
    def thresholds(self) -> SPLThresholds:
        return self._theta

    # ── Core submission (probabilistic layer) ─────────────────────────────────

    def submit(
        self,
        projection: SemanticProjection,
        k: int = 3,
    ) -> SPLResult:
        """
        Submit a SemanticProjection to the emission engine (E0–E3).

        This is the probabilistic layer: it returns SPLResult with
        ClaimCandidates. No ClaimNode is produced here.

        To cross the protocol boundary, call emit_claim_nodes(result.candidates).
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
        Submit projections from two independent builders for the same unit.

        Applies E4 (JSD check) before individual E0–E3 evaluation.
        If JSD > τ₄, both are BRANCH_CANDIDATE — protocol must decide.

        Raises SPLGatewayError if unit_id differs between projections.
        """
        if proj_alpha.unit_id != proj_beta.unit_id:
            raise SPLGatewayError(
                "submit_dual requires both projections to share the same unit_id. "
                f"Got alpha={proj_alpha.unit_id!r}, beta={proj_beta.unit_id!r}."
            )

        self._validate_projection(proj_alpha)
        self._validate_projection(proj_beta)

        jsd = self._engine.apply_e4(proj_alpha, proj_beta)
        branched = jsd > self._theta.tau_4

        if branched:
            result_alpha = self._make_branch_result(proj_alpha)
            result_beta  = self._make_branch_result(proj_beta)
            self._record(result_alpha)
            self._record(result_beta)
        else:
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
            "event":     "submit_dual",
            "dual_id":   dual.dual_id,
            "unit_id":   dual.unit_id,
            "jsd":       round(jsd, 6),
            "branched":  branched,
            "tau_4":     self._theta.tau_4,
            "timestamp": dual.submitted_at,
        })
        return dual

    def submit_batch(
        self,
        projections: list[SemanticProjection],
        k: int = 3,
    ) -> list[SPLResult]:
        """Submit multiple projections. Order is preserved."""
        return [self.submit(p, k=k) for p in projections]

    # ── Protocol boundary (deterministic layer) ───────────────────────────────

    def emit_claim_nodes(
        self,
        candidates: list[ClaimCandidate],
        jsd: float | None = None,
        evidence_count: int = 1,
        extra_assumptions: list[str] | None = None,
    ) -> list:
        """
        THE ONLY LEGAL PATH from ClaimCandidate to ClaimNode.

        Validates each candidate against the gateway criteria, converts
        to ClaimNode, validates the node structurally, assigns a
        deterministic claim_id, and logs a GatewayEvent.

        Validation per candidate (reject → log → skip):
            1. emission_rule ∈ {E1, E2}          (EMIT condition)
            2. relation_score ≥ τ₁               (confidence — E1 only)
            3. h_norm < τ₂                        (entropy ceiling — E1)
               h_norm < τ₃                        (entropy ceiling — E2)
            4. jsd ≤ τ₄ (if provided)             (builder divergence)
            5. evidence_count ≥ MIN_EVIDENCE       (evidence floor)

        After conversion:
            6. validate_claim_node(node)           (structural completeness)
            7. node.claim_id = hash_claim(...)     (deterministic identity)

        Parameters
        ----------
        candidates        List of ClaimCandidates from a READY_FOR_CLAIM result.
        jsd               JSD from a dual-builder workflow (optional).
        evidence_count    Number of independent evidence sources (default 1).
        extra_assumptions Additional assumptions to attach to each ClaimNode.

        Returns
        -------
        List of valid ClaimNodes (may be shorter than input if some rejected).
        """
        nodes = []
        for candidate in candidates:
            try:
                self._validate_candidate(candidate, jsd, evidence_count)
                node = self._converter.convert(candidate, extra_assumptions)
                validate_claim_node(node)
                node.claim_id = hash_claim(node.subject, node.predicate, node.object)
                nodes.append(node)
                self._emit_event(candidate, "EMITTED", "", node.claim_id)
            except ClaimValidationError as e:
                self._emit_event(candidate, "REJECTED", f"ClaimValidationError: {e}", "")
            except CandidateRejectedError as e:
                self._emit_event(candidate, "REJECTED", f"CandidateRejectedError: {e}", "")
            except ValueError as e:
                self._emit_event(candidate, "REJECTED", f"ValueError: {e}", "")

        self._log.append({
            "event":       "emit_claim_nodes",
            "input_count": len(candidates),
            "emitted":     len(nodes),
            "rejected":    len(candidates) - len(nodes),
            "timestamp":   time.time(),
        })
        return nodes

    def emit_claims_from_results(
        self,
        results: list[SPLResult],
        extra_assumptions: list[str] | None = None,
    ) -> list:
        """
        Batch-emit ClaimNodes from a list of SPLResults.

        READY_FOR_CLAIM results are passed to emit_claim_nodes().
        Blocked/branched results are skipped and logged.
        """
        all_nodes = []
        for result in results:
            if result.is_ready():
                all_nodes.extend(
                    self.emit_claim_nodes(
                        result.candidates,
                        extra_assumptions=extra_assumptions,
                    )
                )
            else:
                self._log.append({
                    "event":     "emit_claims_skip",
                    "result_id": result.result_id,
                    "status":    result.status.value,
                    "timestamp": time.time(),
                })
        return all_nodes

    # ── Legacy aliases (route through emit_claim_nodes) ───────────────────────

    def to_claims(
        self,
        result: SPLResult,
        extra_assumptions: list[str] | None = None,
    ) -> list:
        """
        Convert a READY_FOR_CLAIM SPLResult to ClaimNodes.

        Raises SPLGatewayError if result.status is not READY_FOR_CLAIM.
        Internally calls emit_claim_nodes().
        """
        if not result.is_ready():
            raise SPLGatewayError(
                f"Cannot convert result with status={result.status.value} to ClaimNodes. "
                f"Only READY_FOR_CLAIM results are convertible. "
                f"(result_id={result.result_id}, unit_id={result.unit_id})"
            )
        nodes = self.emit_claim_nodes(
            result.candidates, extra_assumptions=extra_assumptions
        )
        self._log.append({
            "event":       "to_claims",
            "result_id":   result.result_id,
            "unit_id":     result.unit_id,
            "claim_count": len(nodes),
            "timestamp":   time.time(),
        })
        return nodes

    def to_claims_batch(
        self,
        results: list[SPLResult],
        extra_assumptions: list[str] | None = None,
    ) -> list:
        """Convert a list of READY_FOR_CLAIM results. Skips blocked/branched."""
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
        Return the complete in-memory audit log for this gateway session.

        For persisted GatewayEvents (emit_claim_nodes decisions), see
        the audit_log.json file (JSON Lines format).
        """
        return list(self._log)

    def summary(self) -> dict:
        """Return aggregated statistics for this gateway session."""
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
            "submissions":      sum(statuses.values()),
            "by_status":        statuses,
            "by_emission_rule": rules,
            "total_candidates": total_candidates,
            "total_claims":     total_claims,
            "thresholds": {
                "tau_0": self._theta.tau_0,
                "tau_1": self._theta.tau_1,
                "tau_2": self._theta.tau_2,
                "tau_3": self._theta.tau_3,
                "tau_4": self._theta.tau_4,
            },
        }

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _validate_candidate(
        self,
        candidate: ClaimCandidate,
        jsd: float | None,
        evidence_count: int,
    ) -> None:
        """
        Validate a ClaimCandidate against gateway criteria.

        Raises CandidateRejectedError with a descriptive reason if any
        criterion is not met.
        """
        # 1. Emission rule must be EMIT (E1 or E2)
        if candidate.emission_rule not in (EmissionRule.E1, EmissionRule.E2):
            raise CandidateRejectedError(
                f"emission_rule={candidate.emission_rule.value} is not EMIT. "
                "Only E1/E2 candidates are convertible."
            )

        # 2. Confidence (relation_score ≥ τ₁ for E1; defensive check)
        if (candidate.emission_rule == EmissionRule.E1
                and candidate.relation_score < self._theta.tau_1):
            raise CandidateRejectedError(
                f"E1 confidence={candidate.relation_score:.4f} < τ₁={self._theta.tau_1}"
            )

        # 3. Entropy ceiling
        if candidate.emission_rule == EmissionRule.E1:
            if candidate.h_norm >= self._theta.tau_2:
                raise CandidateRejectedError(
                    f"E1 entropy h_norm={candidate.h_norm:.4f} ≥ τ₂={self._theta.tau_2}"
                )
        else:  # E2
            if candidate.h_norm >= self._theta.tau_3:
                raise CandidateRejectedError(
                    f"E2 entropy h_norm={candidate.h_norm:.4f} ≥ τ₃={self._theta.tau_3}"
                )

        # 4. JSD (if provided from dual-builder workflow)
        if jsd is not None and jsd > self._theta.tau_4:
            raise CandidateRejectedError(
                f"JSD={jsd:.4f} > τ₄={self._theta.tau_4}. "
                "Use submit_dual() for dual-builder projections."
            )

        # 5. Evidence count
        if evidence_count < MIN_EVIDENCE:
            raise CandidateRejectedError(
                f"evidence_count={evidence_count} < MIN_EVIDENCE={MIN_EVIDENCE}"
            )

    def _emit_event(
        self,
        candidate: ClaimCandidate,
        decision: str,
        reason: str,
        claim_id: str,
    ) -> None:
        """Create a GatewayEvent, add to in-memory log, persist to JSON."""
        event = GatewayEvent(
            candidate_id=candidate.candidate_id,
            emission_rule=candidate.emission_rule.value,
            thresholds={
                "tau_0": self._theta.tau_0,
                "tau_1": self._theta.tau_1,
                "tau_2": self._theta.tau_2,
                "tau_3": self._theta.tau_3,
                "tau_4": self._theta.tau_4,
            },
            decision=decision,
            reason=reason,
            claim_id=claim_id,
        )
        self._log.append({"event": "gateway_event", **event.to_dict()})
        self._persist_event(event)

    def _persist_event(self, event: GatewayEvent) -> None:
        """Append a GatewayEvent to audit_log.json (JSON Lines format)."""
        if not self._audit_log_path:
            return
        try:
            with open(self._audit_log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(event.to_dict()) + "\n")
        except OSError as e:
            # Non-fatal: log to in-memory only if file write fails
            self._log.append({
                "event":  "audit_log_write_error",
                "error":  str(e),
                "timestamp": time.time(),
            })

    def _validate_projection(self, projection: SemanticProjection) -> None:
        """Validate P_r structure before emission."""
        if not projection.P_r:
            raise SPLGatewayError(
                f"projection.P_r is empty (projection_id={projection.projection_id}). "
                "NLP backend must provide a non-empty relational distribution."
            )
        total = sum(projection.P_r.values())
        if abs(total - 1.0) > 0.01:
            raise SPLGatewayError(
                f"projection.P_r sums to {total:.4f}, expected 1.0 "
                f"(projection_id={projection.projection_id})."
            )

    def _make_branch_result(self, projection: SemanticProjection) -> SPLResult:
        """Create a BRANCH_CANDIDATE result without re-running emit()."""
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

    def _record(self, result: SPLResult) -> None:
        """Append a submit event to the in-memory audit log."""
        self._log.append({
            "event":           "submit",
            "result_id":       result.result_id,
            "unit_id":         result.unit_id,
            "projection_id":   result.projection_id,
            "status":          result.status.value,
            "emission_rule":   result.emission_rule.value if result.emission_rule else None,
            "h_norm":          round(result.h_norm, 6),
            "candidate_count": len(result.candidates),
            "builder_origin":  result.builder_origin,
            "matrix_version":  result.matrix_version,
            "timestamp":       result.submitted_at,
        })


# ── Convenience factory ───────────────────────────────────────────────────────

def make_gateway(
    tau_0: float = 0.50,
    tau_1: float = 0.60,
    tau_2: float = 0.25,
    tau_3: float = 0.65,
    tau_4: float = 0.40,
    audit_log_path: str | None = "audit_log.json",
) -> SPLGateway:
    """
    Factory for constructing a calibrated SPLGateway.

        gateway = make_gateway(tau_1=0.70, tau_2=0.20)  # stricter E1

    Raises SPLGatewayError if the thresholds are invalid.
    """
    return SPLGateway(
        thresholds=SPLThresholds(
            tau_0=tau_0, tau_1=tau_1,
            tau_2=tau_2, tau_3=tau_3,
            tau_4=tau_4,
        ),
        audit_log_path=audit_log_path,
    )
