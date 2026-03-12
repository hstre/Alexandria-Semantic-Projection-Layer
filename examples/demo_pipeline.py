"""
Alexandria SPL — Pipeline Demo
===============================
Demonstrates the three-layer architecture end-to-end:

    Text  →  [spl.py]  →  SPLResult  →  [spl_gateway]  →  ClaimNodes

No external dependencies. Runs standalone.

Usage:
    python examples/demo_pipeline.py
"""

import sys
import os
import uuid

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import tests._mock_schema as _mock
_mock.setup()

from spl import SemanticProjection, SPLThresholds
from spl_gateway import SPLGateway


# ── helpers ───────────────────────────────────────────────────────────────────

def _proj(P_r, subjects, objects, builder="alpha", unit_id=None, p_illegal=0.0):
    return SemanticProjection(
        projection_id=str(uuid.uuid4()),
        unit_id=unit_id or str(uuid.uuid4()),
        builder_origin=builder,
        matrix_version="v2.2.0-DEMO",
        P_r=P_r,
        subject_candidates=list(subjects),
        object_candidates=list(objects),
        P_modality={"asserted": 1.0},
        P_category={"dynamic": 1.0},
        p_illegal=p_illegal,
    )

def hr(char="-", w=62): print(char * w)
def section(title):
    print(); hr("═"); print(f"  {title}"); hr("═")


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    θ  = SPLThresholds()
    gw = SPLGateway(θ)

    section("Alexandria SPL · Pipeline Demo")
    print(f"  Thresholds: τ₀={θ.tau_0} τ₁={θ.tau_1} τ₂={θ.tau_2} τ₃={θ.tau_3} τ₄={θ.tau_4}")

    # ── E1: dominant single relation ──────────────────────────────────────────
    # E1 fires when: max_prob > τ₁=0.60 AND H_norm < τ₂=0.25
    # P_r = {0.98, 0.02} → H_norm ≈ 0.14  ✓
    print(); hr()
    print("  E1 — Clear, dominant relation")
    print('  Text: "Paris is the capital of France."')
    hr()
    result = gw.submit(_proj(
        P_r={"IS_CAPITAL_OF": 0.98, "LOCATED_IN": 0.02},
        subjects=["Paris"], objects=["France"],
    ))
    print(f"  Rule: {result.emission_rule.name}  |  H_norm={result.h_norm:.3f}  "
          f"|  Status: {result.status.name}")
    nodes = gw.emit_claim_nodes(result.candidates)
    for n in nodes:
        print(f"  → ClaimNode: {n.subject!r} --[{n.predicate}]--> {n.object!r}")
        print(f"    claim_id : {n.claim_id[:20]}…")

    # ── E2: multiple relations above τ₂ ──────────────────────────────────────
    # E2 fires when: H_norm ∈ [τ₂, τ₃) = [0.25, 0.65)
    # P_r = {0.80, 0.15, 0.05} → H_norm ≈ 0.56  ✓
    print(); hr()
    print("  E2 — Multiple relations (top-k above τ₂)")
    print('  Text: "Remote work largely increases, and to some degree changes, productivity."')
    hr()
    result = gw.submit(_proj(
        P_r={"INCREASES": 0.80, "MODIFIES": 0.15, "CORRELATES_WITH": 0.05},
        subjects=["remote_work"], objects=["productivity"],
    ))
    print(f"  Rule: {result.emission_rule.name}  |  H_norm={result.h_norm:.3f}  "
          f"|  Status: {result.status.name}")
    nodes = gw.emit_claim_nodes(result.candidates)
    for n in nodes:
        print(f"  → ClaimNode: {n.subject!r} --[{n.predicate}]--> {n.object!r}")

    # ── E3: high entropy → gateway blocks ────────────────────────────────────
    # E3 fires when: H_norm ≥ τ₃=0.65
    # P_r = {0.35, 0.35, 0.30} → H_norm ≈ 1.00  ✓
    print(); hr()
    print("  E3 — High entropy → AMBIGUOUS (gateway blocks)")
    print('  Text: "The effect may or may not reduce, increase or alter the outcome."')
    hr()
    result = gw.submit(_proj(
        P_r={"INCREASES": 0.35, "DECREASES": 0.35, "CORRELATES_WITH": 0.30},
        subjects=["effect"], objects=["outcome"],
    ))
    print(f"  Rule: {result.emission_rule.name}  |  H_norm={result.h_norm:.3f}  "
          f"|  Status: {result.status.name}")
    nodes = gw.emit_claim_nodes(result.candidates)
    print(f"  → Gateway blocked: {len(nodes)} ClaimNodes emitted  "
          f"(requires human review or re-projection)")

    # ── E4: builder divergence → gateway blocks ───────────────────────────────
    # E4 fires when: JSD(P_alpha, P_beta) > τ₄=0.40
    print(); hr()
    print("  E4 — Builder divergence → BRANCH_CANDIDATE (gateway blocks)")
    print('  Alpha: "Policy X causes outcome Y"  |  '
          'Beta: "Policy X prevents outcome Y"')
    hr()
    uid = str(uuid.uuid4())
    dual = gw.submit_dual(
        _proj({"CAUSES": 0.98, "CORRELATES_WITH": 0.02},
              ["policy_x"], ["outcome_y"], builder="alpha", unit_id=uid),
        _proj({"PREVENTS": 0.97, "CORRELATES_WITH": 0.03},
              ["policy_x"], ["outcome_y"], builder="beta",  unit_id=uid),
    )
    print(f"  JSD={dual.jsd:.3f}  (τ₄={θ.tau_4})  |  "
          f"Alpha: {dual.result_alpha.status.name}  |  "
          f"Beta:  {dual.result_beta.status.name}")
    nodes_a = gw.emit_claim_nodes(dual.result_alpha.candidates, jsd=dual.jsd)
    nodes_b = gw.emit_claim_nodes(dual.result_beta.candidates,  jsd=dual.jsd)
    print(f"  → Gateway blocked: {len(nodes_a) + len(nodes_b)} ClaimNodes emitted  "
          f"(branch must be resolved at protocol level)")

    # ── summary ───────────────────────────────────────────────────────────────
    print(); hr("═")
    s = gw.summary()
    blocked = s["by_status"].get("ambiguous", 0) + s["by_status"].get("branch_candidate", 0)
    ready   = s["by_status"].get("ready_for_claim", 0)
    print(f"  Gateway summary ·  submissions={s['submissions']}  "
          f"ready={ready}  blocked={blocked}")
    print(f"  Rules fired    ·  {s['by_emission_rule']}")
    hr("═"); print()


if __name__ == "__main__":
    main()
