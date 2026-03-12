"""
Protocol boundary: ClaimCandidate → ClaimNode conversion (WP2 §2, §7.4).

_CATEGORY_HINT_MAP              — SPL hint → Alexandria Category mapping
_MODALITY_HINT_MAP              — SPL hint → Alexandria Modality mapping
validate_candidate_for_protocol_entry() — explicit pre-conversion boundary check
ClaimCandidateConverter         — the ONLY legal ClaimCandidate→ClaimNode path

These live in spl_gateway (not spl.py) because they encode Alexandria protocol
semantics. The SPL provides *hints* — this layer translates them into canonical
protocol types.
"""

from __future__ import annotations

from spl import ClaimCandidate, EmissionRule, SPLThresholds
from spl_gateway._exceptions import CandidateRejectedError


# ── Protocol semantic mappings ────────────────────────────────────────────────

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


# ── Explicit boundary validation ──────────────────────────────────────────────

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


# ── ClaimCandidateConverter ───────────────────────────────────────────────────

class ClaimCandidateConverter:
    """
    The ONLY legal path from ClaimCandidate to ClaimNode.

    This class enforces the protocol boundary defined in WP2 §2:
    "The epistemic validity of all claims remains governed exclusively
    by the Alexandria protocol."

    Lives in spl_gateway (not spl.py) because it encodes protocol semantics:
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
