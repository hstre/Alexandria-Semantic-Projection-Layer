"""
SPLGateway — the formal protocol entry point (WP2 §2).

SPLGateway   — main class, instantiate once per session
make_gateway — convenience factory
MIN_EVIDENCE — minimum evidence count constant
"""

from __future__ import annotations

import json
import os
import time
import uuid

from spl import EmissionRule, EmissionStatus, SemanticProjection, SPLThresholds, EmissionEngine, compute_jsd

from spl_gateway._exceptions import (
    SPLGatewayError,
    CandidateRejectedError,
    ClaimValidationError,
)
from spl_gateway._types import GatewayEvent, SPLResult, DualBuilderResult
from spl_gateway._utils import hash_claim, validate_claim_node
from spl_gateway._converter import ClaimCandidateConverter


# ── Constants ─────────────────────────────────────────────────────────────────

MIN_EVIDENCE: int = 1
"""Minimum evidence count for a candidate to pass gateway validation."""


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
        candidates,
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
        candidate,
        jsd: float | None,
        evidence_count: int,
    ) -> None:
        """Validate a ClaimCandidate against gateway criteria."""
        if candidate.emission_rule not in (EmissionRule.E1, EmissionRule.E2):
            raise CandidateRejectedError(
                f"emission_rule={candidate.emission_rule.value} is not EMIT. "
                "Only E1/E2 candidates are convertible."
            )

        if (candidate.emission_rule == EmissionRule.E1
                and candidate.relation_score < self._theta.tau_1):
            raise CandidateRejectedError(
                f"E1 confidence={candidate.relation_score:.4f} < τ₁={self._theta.tau_1}"
            )

        if candidate.emission_rule == EmissionRule.E1:
            if candidate.h_norm >= self._theta.tau_2:
                raise CandidateRejectedError(
                    f"E1 entropy h_norm={candidate.h_norm:.4f} ≥ τ₂={self._theta.tau_2}"
                )
        else:
            if candidate.h_norm >= self._theta.tau_3:
                raise CandidateRejectedError(
                    f"E2 entropy h_norm={candidate.h_norm:.4f} ≥ τ₃={self._theta.tau_3}"
                )

        if jsd is not None and jsd > self._theta.tau_4:
            raise CandidateRejectedError(
                f"JSD={jsd:.4f} > τ₄={self._theta.tau_4}. "
                "Use submit_dual() for dual-builder projections."
            )

        if evidence_count < MIN_EVIDENCE:
            raise CandidateRejectedError(
                f"evidence_count={evidence_count} < MIN_EVIDENCE={MIN_EVIDENCE}"
            )

    def _emit_event(
        self,
        candidate,
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
