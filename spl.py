"""
Alexandria Core — spl.py
Semantic Projection Layer (SPL) — Probabilistic Pre-Protocol Stage
==================================================================

Working Paper 2: "Semantic Projection Layer — A Formal Bridge between
Natural Language and Epistemic Protocol" (Rentschler, v16)

This module implements the PROBABILISTIC stage of the SPL pipeline.
It does NOT implement the protocol boundary — that is spl_gateway.py.

What this module provides:
    - SemanticUnit       minimal epistemic text fragment
    - SemanticProjection probabilistic relational structure π(s)
    - ClaimCandidate     discrete triple extracted by an emission rule
    - EmissionStatus     PROJECTED / READY_FOR_CLAIM / AMBIGUOUS /
                         BRANCH_CANDIDATE / STRUCTURAL_VIOLATION
    - EmissionRule       E0 / E1 / E2 / E3 / E4
    - SPLThresholds      Θ = {τ₀, τ₁, τ₂, τ₃, τ₄}
    - EmissionEngine     evaluates E0–E4 against a SemanticProjection
    - compute_h_norm()   normalised Shannon entropy (WP2 §7.1)
    - compute_jsd()      Jensen-Shannon Divergence (WP2 §3.3.5)

What this module does NOT provide:
    - ClaimCandidateConverter   → spl_gateway.py
    - Protocol boundary logic   → spl_gateway.py
    - ClaimNode creation        → spl_gateway.py

Protocol invariant [SHALL]:
    No text fragment may become a canonical ClaimNode directly.
    The path is: text → SemanticUnit → SemanticProjection →
                 ClaimCandidate → (spl_gateway.py) → ClaimNode

Reference: WP2 §3, §7, Appendix I
"""

from __future__ import annotations

import math
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


# ── Emission Status (WP2 §7.2, Appendix I.1) ─────────────────────────────────

class EmissionStatus(str, Enum):
    """
    Status of a SemanticProjection after emission rule evaluation.

    WP2 Appendix I.1 defines five statuses:
        PROJECTED        — projection computed, emission not yet evaluated
        READY_FOR_CLAIM  — E1 or E2: candidate(s) emitted, eligible for protocol
        AMBIGUOUS        — E3: H_norm >= τ₃, no emission (projection too uncertain)
        BRANCH_CANDIDATE — E4: JSD(A,B) > τ₄, dual-builder divergence too large
        STRUCTURAL_VIOLATION — E0: P_illegal > τ₀, ontological shield triggered
    """
    PROJECTED             = "projected"
    READY_FOR_CLAIM       = "ready_for_claim"
    AMBIGUOUS             = "ambiguous"
    BRANCH_CANDIDATE      = "branch_candidate"
    STRUCTURAL_VIOLATION  = "structural_violation"


class EmissionRule(str, Enum):
    """Which emission rule produced this result. WP2 §7.2."""
    E0 = "E0"   # Structural rejection
    E1 = "E1"   # Singular emission (argmax)
    E2 = "E2"   # Multiple emission (top-k)
    E3 = "E3"   # Ambiguity block
    E4 = "E4"   # Branch on builder divergence


# ── Thresholds Θ (WP2 §7.2, Appendix I.1) ────────────────────────────────────

@dataclass
class SPLThresholds:
    """
    Parameter set Θ = {τ₀, τ₁, τ₂, τ₃, τ₄}.

    WP2 Appendix I.1 recommended initial values:
        τ₀ ≈ 0.50  structural rejection threshold
        τ₁ ≈ 0.60  singular emission: max(P_r) must exceed this
        τ₂ ≈ 0.25  singular emission: H_norm must be below this
        τ₃ ≈ 0.65  ambiguity block threshold
        τ₄ ≈ 0.40  builder divergence branch threshold

    [HEURISTIC] These are genesis axiom defaults per WP2 §3.6.5.
    They should be calibrated against domain-specific gold standards
    through the governance cycle (WP2 Appendix J).

    The thresholds partition the simplex Π into geometric regions
    (WP2 §3.4.2):
        H_norm < τ₂         → vertex neighbourhood (E1)
        τ₂ ≤ H_norm < τ₃   → face region (E2)
        H_norm ≥ τ₃         → interior/centroid (E3 block)
    """
    tau_0: float = 0.50   # structural rejection
    tau_1: float = 0.60   # singular emission dominance
    tau_2: float = 0.25   # singular emission entropy ceiling
    tau_3: float = 0.65   # ambiguity block floor
    tau_4: float = 0.40   # builder divergence branch threshold

    def validate(self) -> list[str]:
        errors = []
        if not (0 < self.tau_0 < 1):
            errors.append(f"tau_0={self.tau_0} must be in (0,1)")
        if not (0 < self.tau_1 < 1):
            errors.append(f"tau_1={self.tau_1} must be in (0,1)")
        if not (0 < self.tau_2 < self.tau_3 <= 1):
            errors.append(f"tau_2={self.tau_2} must be < tau_3={self.tau_3}")
        if not (0 < self.tau_4 < 1):
            errors.append(f"tau_4={self.tau_4} must be in (0,1)")
        return errors


# ── SemanticUnit (WP2 §3.1) ──────────────────────────────────────────────────

@dataclass
class SemanticUnit:
    """
    The smallest extractable text fragment that can carry a relational
    epistemic assertion. WP2 §3.1:

        u ∈ U iff it can be projected onto an interpretable relational
        structure with at least one subject candidate, one relation
        candidate, and one object candidate.

    A single sentence may contain multiple SemanticUnits.

    Fields
    ------
    unit_id        Unique identifier
    source_text    The raw text fragment
    source_ref     Document/work identifier of origin
    offset_start   Character offset in source document
    offset_end     Character offset end
    fragmentation_signal  What boundary signal triggered extraction
                          (relational verb, modal, evidential, etc.)
    created_at     Timestamp
    """
    unit_id:              str
    source_text:          str
    source_ref:           str
    offset_start:         int = 0
    offset_end:           int = 0
    fragmentation_signal: str = ""
    created_at:           float = field(default_factory=time.time)

    @classmethod
    def new(cls, source_text: str, source_ref: str,
            offset_start: int = 0, offset_end: int = 0,
            fragmentation_signal: str = "") -> "SemanticUnit":
        return cls(
            unit_id=str(uuid.uuid4()),
            source_text=source_text,
            source_ref=source_ref,
            offset_start=offset_start,
            offset_end=offset_end,
            fragmentation_signal=fragmentation_signal,
        )

    def to_dict(self) -> dict:
        return {
            "unit_id":              self.unit_id,
            "source_text":          self.source_text,
            "source_ref":           self.source_ref,
            "offset_start":         self.offset_start,
            "offset_end":           self.offset_end,
            "fragmentation_signal": self.fragmentation_signal,
            "created_at":           self.created_at,
        }


# ── SemanticProjection (WP2 §3.3) ────────────────────────────────────────────

@dataclass
class SemanticProjection:
    """
    The output of π(s): a probabilistic relational structure over the
    constrained relation space ℛ. WP2 §3.3.1:

        π(s) = (E_s, P_r, E_o, P_c, P_m, P_scope, U)

    The full distributional information is ALWAYS retained here —
    no information is discarded at emission (WP2 §3.3.4).

    Fields
    ------
    projection_id      Unique identifier
    unit_id            Back-reference to originating SemanticUnit
    builder_origin     "alpha" | "beta"
    matrix_version     Version of relation matrix M used (WP2 Appendix K)

    P_r                Relational distribution over ℛ — dict {relation: prob}
                       Must sum to 1.0. This is the marginal of tensor R.
    subject_candidates List of subject entity strings (E_s)
    object_candidates  List of object entity strings (E_o)
    P_category         Distribution over semantic category hints
                       NOTE [DBA]: These are SEMANTIC HINTS, not epistemic
                       categories. The protocol sets category on ClaimNode.
                       Key: "ontic"|"dynamic"|"statistical"|"epistemic"|
                            "model"|"normative" — NOT Alexandria's Category enum.
    P_modality         Distribution over modality hints
                       Key: "asserted"|"suggested"|"hypothesized"|"possible"
    P_scope            Distribution over scope dimensions (optional)

    h_norm             Normalised Shannon entropy H_norm ∈ [0,1] (WP2 §7.1)
    status             EmissionStatus after E0–E4 evaluation
    emission_rule      Which rule fired
    matrix_seal_hash   Cryptographic hash of matrix version (WP2 Appendix K.6)
    created_at         Timestamp
    """
    projection_id:      str
    unit_id:            str
    builder_origin:     str       # "alpha" | "beta"
    matrix_version:     str       # e.g. "v2.2.0-SML"

    # Core distributional outputs
    P_r:                dict[str, float]       # relational distribution
    subject_candidates: list[str]              # E_s
    object_candidates:  list[str]              # E_o

    # Secondary distributions (hints only — not protocol categories)
    P_category:         dict[str, float] = field(default_factory=dict)
    P_modality:         dict[str, float] = field(default_factory=dict)
    P_scope:            dict[str, float] = field(default_factory=dict)

    # Derived metrics
    h_norm:             float = 0.0
    status:             EmissionStatus = EmissionStatus.PROJECTED
    emission_rule:      Optional[EmissionRule] = None
    p_illegal:          float = 0.0  # for E0 audit

    # Governance
    matrix_seal_hash:   str = ""

    created_at:         float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        return {
            "projection_id":      self.projection_id,
            "unit_id":            self.unit_id,
            "builder_origin":     self.builder_origin,
            "matrix_version":     self.matrix_version,
            "P_r":                self.P_r,
            "subject_candidates": self.subject_candidates,
            "object_candidates":  self.object_candidates,
            "P_category":         self.P_category,
            "P_modality":         self.P_modality,
            "h_norm":             self.h_norm,
            "status":             self.status.value,
            "emission_rule":      self.emission_rule.value if self.emission_rule else None,
            "matrix_seal_hash":   self.matrix_seal_hash,
            "created_at":         self.created_at,
        }


# ── ClaimCandidate (WP2 §3.3.4, §7.2) ───────────────────────────────────────

@dataclass
class ClaimCandidate:
    """
    A discrete relational triple extracted from a SemanticProjection
    by one of the emission rules E1 or E2. WP2 §3.3.4:

        C = (s*, r*, o*) ∈ E_s × ℛ × E_o

    A ClaimCandidate is NOT a canonical claim. It becomes a ClaimNode
    only through ClaimCandidateConverter, which applies the protocol
    boundary rules.

    Key invariant [SHALL]:
        ClaimCandidate.semantic_category_hint is a HINT, not a protocol
        category. The converter maps it to Alexandria's Category enum.
        The protocol — not the SPL — makes the final epistemic classification.

    Fields
    ------
    candidate_id       Unique identifier
    projection_id      Back-reference to originating SemanticProjection
    unit_id            Back-reference to originating SemanticUnit (for C⁺)
    source_ref         Document reference

    subject            Best subject entity (s*)
    relation           Best relation (r*) — from ℛ
    object             Best object entity (o*)

    relation_score     R(s*, r*, o*) — probability mass of this triple
    rank               1 for E1, 1..k for E2
    emission_rule      E1 | E2

    modality_hint      Dominant modality from P_m
                       ("asserted"|"suggested"|"hypothesized"|"possible")
    scope_hint         Scope from P_scope (optional)
    semantic_category_hint  Dominant semantic category hint
                       NOT an Alexandria Category — see above
    h_norm             H_norm of originating projection (provenance)
    matrix_version     Matrix version active at emission
    builder_origin     "alpha" | "beta"
    created_at         Timestamp
    """
    candidate_id:              str
    projection_id:             str
    unit_id:                   str
    source_ref:                str

    # The triple
    subject:                   str
    relation:                  str
    object:                    str

    # Emission metadata
    relation_score:            float     # R(s*, r*, o*)
    rank:                      int = 1   # 1 = top (E1), >1 = E2
    emission_rule:             EmissionRule = EmissionRule.E1

    # Inherited from projection
    modality_hint:             str = "asserted"
    scope_hint:                str = ""
    semantic_category_hint:    str = ""   # "dynamic"|"statistical"|etc. — NOT Category enum
    h_norm:                    float = 0.0
    matrix_version:            str = ""
    builder_origin:            str = "alpha"

    created_at:                float = field(default_factory=time.time)

    @classmethod
    def new(
        cls,
        projection: SemanticProjection,
        subject: str,
        relation: str,
        object_: str,
        relation_score: float,
        rank: int = 1,
        emission_rule: EmissionRule = EmissionRule.E1,
    ) -> "ClaimCandidate":
        return cls(
            candidate_id=str(uuid.uuid4()),
            projection_id=projection.projection_id,
            unit_id=projection.unit_id,
            source_ref="",
            subject=subject,
            relation=relation,
            object=object_,
            relation_score=relation_score,
            rank=rank,
            emission_rule=emission_rule,
            modality_hint=_dominant(projection.P_modality) or "asserted",
            semantic_category_hint=_dominant(projection.P_category) or "",
            h_norm=projection.h_norm,
            matrix_version=projection.matrix_version,
            builder_origin=projection.builder_origin,
        )

    def to_dict(self) -> dict:
        return {
            "candidate_id":           self.candidate_id,
            "projection_id":          self.projection_id,
            "unit_id":                self.unit_id,
            "source_ref":             self.source_ref,
            "subject":                self.subject,
            "relation":               self.relation,
            "object":                 self.object,
            "relation_score":         self.relation_score,
            "rank":                   self.rank,
            "emission_rule":          self.emission_rule.value,
            "modality_hint":          self.modality_hint,
            "semantic_category_hint": self.semantic_category_hint,
            "h_norm":                 self.h_norm,
            "matrix_version":         self.matrix_version,
            "builder_origin":         self.builder_origin,
            "created_at":             self.created_at,
        }


# ── JSD computation (WP2 §3.3.5) ─────────────────────────────────────────────

def compute_jsd(p: dict[str, float], q: dict[str, float]) -> float:
    """
    Jensen-Shannon Divergence between two relational distributions.
    Base-2 logarithm → JSD ∈ [0, 1]. WP2 §3.3.5.

    Global simplex embedding: the union of all keys is used as ℛ_global.
    Missing keys are assigned probability 0 (zero-padding per WP2 §3.3.5).

    If ℛ_A ∩ ℛ_B = ∅, JSD = 1.0 by construction (WP2 §3.3.5):
    "a fundamental ontological disagreement between builders
    maximises JSD by construction."
    """
    all_keys = set(p) | set(q)
    if not all_keys:
        return 0.0

    pv = {k: p.get(k, 0.0) for k in all_keys}
    qv = {k: q.get(k, 0.0) for k in all_keys}

    m = {k: 0.5 * (pv[k] + qv[k]) for k in all_keys}

    def kl(a: dict, b: dict) -> float:
        s = 0.0
        for k in all_keys:
            if a[k] > 0 and b[k] > 0:
                s += a[k] * math.log2(a[k] / b[k])
        return s

    return 0.5 * kl(pv, m) + 0.5 * kl(qv, m)


def compute_h_norm(P_r: dict[str, float]) -> float:
    """
    Normalised Shannon entropy of a relational distribution. WP2 §7.1:
        H_norm = H(P_r) / log(|ℛ|)  ∈ [0, 1]
    """
    n = len(P_r)
    if n <= 1:
        return 0.0
    h = 0.0
    for p in P_r.values():
        if p > 0:
            h -= p * math.log2(p)
    return h / math.log2(n)


# ── Emission engine (WP2 §7.2, Appendix I.1) ─────────────────────────────────

class EmissionEngine:
    """
    Evaluates emission rules E0–E4 against a SemanticProjection.

    E0 — Structural rejection: P_illegal > τ₀ → STRUCTURAL_VIOLATION
    E1 — Singular: max(P_r) > τ₁ AND H_norm < τ₂ → single ClaimCandidate
    E2 — Multiple: max(P_r) ≤ τ₁ AND H_norm < τ₃ → top-k ClaimCandidates
    E3 — Block: H_norm ≥ τ₃ → AMBIGUOUS, no emission
    E4 — Branch: JSD(A, B) > τ₄ → BRANCH_CANDIDATE (dual-builder only)

    [HEURISTIC] Default k=3 for E2. Override via emit(k=...).
    """

    def __init__(self, thresholds: SPLThresholds | None = None):
        self.Θ = thresholds or SPLThresholds()

    def emit(
        self,
        projection: SemanticProjection,
        k: int = 3,
    ) -> list[ClaimCandidate]:
        """
        Apply E0–E3 to a single projection.
        Returns list of ClaimCandidates (empty for E3/E0).
        Updates projection.status and projection.emission_rule in place.
        """
        # E0 — structural rejection
        if projection.p_illegal > self.Θ.tau_0:
            projection.status = EmissionStatus.STRUCTURAL_VIOLATION
            projection.emission_rule = EmissionRule.E0
            return []

        P_r = projection.P_r
        if not P_r:
            projection.status = EmissionStatus.AMBIGUOUS
            projection.emission_rule = EmissionRule.E3
            return []

        h = compute_h_norm(P_r)
        projection.h_norm = h

        max_rel = max(P_r, key=P_r.get)
        max_prob = P_r[max_rel]

        # E3 — block (checked before E1/E2)
        if h >= self.Θ.tau_3:
            projection.status = EmissionStatus.AMBIGUOUS
            projection.emission_rule = EmissionRule.E3
            return []

        # E1 — singular
        if max_prob > self.Θ.tau_1 and h < self.Θ.tau_2:
            projection.status = EmissionStatus.READY_FOR_CLAIM
            projection.emission_rule = EmissionRule.E1
            subj = projection.subject_candidates[0] if projection.subject_candidates else ""
            obj  = projection.object_candidates[0]  if projection.object_candidates  else ""
            return [ClaimCandidate.new(projection, subj, max_rel, obj, max_prob, rank=1,
                                       emission_rule=EmissionRule.E1)]

        # E2 — multiple
        projection.status = EmissionStatus.READY_FOR_CLAIM
        projection.emission_rule = EmissionRule.E2
        top_k = sorted(P_r.items(), key=lambda x: -x[1])[:k]
        subj = projection.subject_candidates[0] if projection.subject_candidates else ""
        obj  = projection.object_candidates[0]  if projection.object_candidates  else ""
        return [
            ClaimCandidate.new(projection, subj, rel, obj, prob, rank=i+1,
                               emission_rule=EmissionRule.E2)
            for i, (rel, prob) in enumerate(top_k)
        ]

    def apply_e4(
        self,
        proj_a: SemanticProjection,
        proj_b: SemanticProjection,
    ) -> float:
        """
        Compute JSD between two builders' projections of the same SemanticUnit.
        If JSD > τ₄, marks both projections as BRANCH_CANDIDATE.
        Returns JSD value.
        """
        jsd = compute_jsd(proj_a.P_r, proj_b.P_r)
        if jsd > self.Θ.tau_4:
            proj_a.status = EmissionStatus.BRANCH_CANDIDATE
            proj_b.status = EmissionStatus.BRANCH_CANDIDATE
            proj_a.emission_rule = EmissionRule.E4
            proj_b.emission_rule = EmissionRule.E4
        return jsd


# ── Helpers ───────────────────────────────────────────────────────────────────

def _dominant(dist: dict[str, float]) -> str:
    """Return argmax of a distribution dict, or '' if empty."""
    if not dist:
        return ""
    return max(dist, key=dist.get)


# ── NLP convenience entry point ───────────────────────────────────────────────

def project_text(text: str, **kwargs) -> "SemanticProjection":
    """
    Convenience function: text → SemanticProjection via NLP backend.

    Requires ``nlp_backend`` (sentence-transformers) to be installed.
    If nlp_backend is not available an ImportError is raised with a helpful
    message so callers can degrade gracefully.

    Usage::

        from spl import project_text
        proj = project_text("Paris is the capital of France.")
        # proj.P_r → {"capital_of": 0.68, ...}

    All keyword arguments are forwarded to :class:`nlp_backend.SPLNLPBackend`.
    """
    try:
        from nlp_backend import SPLNLPBackend  # optional dependency
    except ImportError as exc:
        raise ImportError(
            "nlp_backend requires sentence-transformers. "
            "Install with: pip install sentence-transformers"
        ) from exc
    backend = SPLNLPBackend(**kwargs)
    return backend.project_text(text)
