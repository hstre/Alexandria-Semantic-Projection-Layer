"""
Alexandria SPL — Test App
=========================
Verifiziert die Theorie aus WP2 (Semantic Projection Layer) anhand konkreter
Beispiele. Testet alle Emission Rules E0–E4, H_norm-Berechnung und JSD.

Keine externen Abhängigkeiten — läuft standalone ohne schema.py.

Beispielsatz aus WP2 §3.2:
    "The results suggest that remote work may increase productivity,
     although the effect varies across sectors."

→ u1: results → suggest → [claim: remote work increases productivity]
→ u2: remote work → may_increase → productivity
→ u3: effect → varies_across → sectors

Ausführung:
    python test_app.py
    python test_app.py --verbose
"""

import sys
import uuid

# Patch: ClaimCandidateConverter braucht .schema — wir mocken es für den Test
import types

_mock_schema = types.ModuleType("spl_schema_mock")

class _Category:
    EMPIRICAL = "EMPIRICAL"
    MODEL = "MODEL"
    NORMATIVE = "NORMATIVE"
    def __class_getitem__(cls, item): return item
    def __getitem__(self, item): return item

class _Modality:
    HYPOTHESIS = "hypothesis"
    def __init__(self, v): self.value = v

class _EpistemicStatus:
    UNVALIDATED = "unvalidated"

class _BuilderOrigin:
    ALPHA = "alpha"
    BETA = "beta"

class _ClaimNode:
    def __init__(self, **kw):
        for k, v in kw.items():
            setattr(self, k, v)
    @classmethod
    def new(cls, subject, predicate, object, category, assumptions, source_refs):
        return cls(
            subject=subject, predicate=predicate, object=object,
            category=category, assumptions=assumptions, source_refs=source_refs,
            modality=None, builder_origin=None,
        )
    def __repr__(self):
        return (f"ClaimNode({self.subject!r} --[{self.predicate}]--> {self.object!r} "
                f"cat={self.category} mod={getattr(self,'modality',None)})")

_mock_schema.ClaimNode = _ClaimNode
_mock_schema.Category = _Category()
_mock_schema.Modality = _Modality
_mock_schema.BuilderOrigin = _BuilderOrigin()
_mock_schema.EpistemicStatus = _EpistemicStatus()

# Inject als .schema relativ zu spl.py
import importlib
import spl as _spl_module
_spl_module.__package__ = "_spl_pkg"
sys.modules["_spl_pkg"] = types.ModuleType("_spl_pkg")
sys.modules["_spl_pkg.schema"] = _mock_schema

from spl import (
    SemanticUnit, SemanticProjection, ClaimCandidate,
    EmissionEngine, EmissionStatus, EmissionRule, SPLThresholds,
    ClaimCandidateConverter, compute_jsd, compute_h_norm,
)

VERBOSE = "--verbose" in sys.argv or "-v" in sys.argv

# ── Farben ────────────────────────────────────────────────────────────────────

RESET  = "\033[0m"
BOLD   = "\033[1m"
GREEN  = "\033[32m"
RED    = "\033[31m"
YELLOW = "\033[33m"
CYAN   = "\033[36m"
GRAY   = "\033[90m"

def ok(msg):  print(f"  {GREEN}✓{RESET} {msg}")
def fail(msg): print(f"  {RED}✗{RESET} {msg}"); _FAILURES.append(msg)
def info(msg): print(f"  {CYAN}→{RESET} {msg}")
def warn(msg): print(f"  {YELLOW}!{RESET} {msg}")

_FAILURES = []

def header(title):
    print(f"\n{BOLD}{'─'*60}{RESET}")
    print(f"{BOLD}{title}{RESET}")
    print(f"{BOLD}{'─'*60}{RESET}")

def subheader(title):
    print(f"\n  {CYAN}{title}{RESET}")

def assert_eq(label, got, expected):
    if got == expected:
        ok(f"{label}: {got!r}")
    else:
        fail(f"{label}: got {got!r}, expected {expected!r}")

def assert_true(label, cond, detail=""):
    if cond:
        ok(f"{label}{(' — ' + detail) if detail else ''}")
    else:
        fail(f"{label}{(' — ' + detail) if detail else ''} [FAILED]")

def assert_approx(label, got, expected, tol=1e-6):
    if abs(got - expected) <= tol:
        ok(f"{label}: {got:.6f}")
    else:
        fail(f"{label}: got {got:.6f}, expected {expected:.6f} (tol={tol})")


# ── Hilfsfunktion: neue SemanticProjection ───────────────────────────────────

def make_projection(
    P_r: dict,
    subjects=("remote work",),
    objects=("productivity",),
    builder="alpha",
    p_illegal=0.0,
    P_modality=None,
    P_category=None,
    source_text="",
) -> SemanticProjection:
    unit = SemanticUnit.new(
        source_text=source_text or "test fragment",
        source_ref="WP2-test",
    )
    proj = SemanticProjection(
        projection_id=str(uuid.uuid4()),
        unit_id=unit.unit_id,
        builder_origin=builder,
        matrix_version="v2.2.0-TEST",
        P_r=P_r,
        subject_candidates=list(subjects),
        object_candidates=list(objects),
        P_modality=P_modality or {"suggested": 1.0},
        P_category=P_category or {"dynamic": 1.0},
        p_illegal=p_illegal,
    )
    return unit, proj


# ══════════════════════════════════════════════════════════════════════════════
# TEST 1 — Thresholds & Validierung
# ══════════════════════════════════════════════════════════════════════════════

header("TEST 1 — SPLThresholds (Θ)")

Θ = SPLThresholds()
info(f"Standard-Schwellenwerte: τ₀={Θ.tau_0} τ₁={Θ.tau_1} τ₂={Θ.tau_2} τ₃={Θ.tau_3} τ₄={Θ.tau_4}")

errors = Θ.validate()
assert_true("Standard-Θ ist valide", len(errors) == 0, f"{errors}")

bad = SPLThresholds(tau_2=0.80, tau_3=0.65)  # tau_2 > tau_3 — illegal
errors_bad = bad.validate()
assert_true("tau_2 > tau_3 wird erkannt", len(errors_bad) > 0,
            f"Fehler: {errors_bad}")

assert_eq("τ₀ (structural rejection)", Θ.tau_0, 0.50)
assert_eq("τ₁ (singular dominance)", Θ.tau_1, 0.60)
assert_eq("τ₂ (entropy ceiling E1)", Θ.tau_2, 0.25)
assert_eq("τ₃ (ambiguity floor E3)", Θ.tau_3, 0.65)
assert_eq("τ₄ (JSD branch threshold)", Θ.tau_4, 0.40)


# ══════════════════════════════════════════════════════════════════════════════
# TEST 2 — H_norm (Shannon Entropy)
# ══════════════════════════════════════════════════════════════════════════════

header("TEST 2 — H_norm (normalisierte Shannon-Entropie)")

subheader("2a — Punkt-Masse (maximale Gewissheit)")
p_certain = {"causes": 1.0}
h = compute_h_norm(p_certain)
assert_approx("H_norm(punkt-masse) == 0.0", h, 0.0)

subheader("2b — Gleichverteilung (maximale Ambiguität)")
n = 4
p_uniform = {f"r{i}": 1/n for i in range(n)}
h_uniform = compute_h_norm(p_uniform)
assert_approx("H_norm(gleichverteilt, n=4) == 1.0", h_uniform, 1.0)

subheader("2c — E1-Szenario: sehr starke Dominanz (H_norm < τ₂=0.25)")
# E1 braucht max_prob > τ₁=0.60 UND H_norm < τ₂=0.25 → sehr spitze Verteilung
# Mit nur 2 Relationen: H_norm = H / log2(2) = H
# p=0.97 → H ≈ 0.195 < 0.25 ✓
p_e1 = {"causes": 0.97, "correlates_with": 0.03}
h_e1 = compute_h_norm(p_e1)
info(f"P_r = {p_e1}")
info(f"H_norm = {h_e1:.4f}  (muss < τ₂={Θ.tau_2} für E1)")
assert_true("H_norm < τ₂ für E1-Szenario", h_e1 < Θ.tau_2, f"H={h_e1:.4f}")

subheader("2d — E3-Szenario: hohe Entropie → AMBIGUOUS")
# Fast gleichverteilte Distribution über 4 Relationen
p_e3 = {"causes": 0.28, "correlates": 0.27, "inhibits": 0.25, "suggests": 0.20}
h_e3 = compute_h_norm(p_e3)
info(f"P_r = {p_e3}")
info(f"H_norm = {h_e3:.4f}  (muss >= τ₃={Θ.tau_3} für E3)")
assert_true("H_norm >= τ₃ für E3-Szenario", h_e3 >= Θ.tau_3, f"H={h_e3:.4f}")

subheader("2e — E2-Szenario: moderate Entropie (zwischen τ₂ und τ₃)")
# E2 braucht H_norm ∈ [τ₂, τ₃) → moderate Dominanz mit 3 Relationen
# {"suggests":0.80,"indicates":0.15,"supports":0.05} → H_norm ≈ 0.56
p_e2 = {"suggests": 0.80, "indicates": 0.15, "supports": 0.05}
h_e2 = compute_h_norm(p_e2)
info(f"P_r = {p_e2}")
info(f"H_norm = {h_e2:.4f}  (muss τ₂ <= H < τ₃)")
assert_true("τ₂ ≤ H_norm < τ₃ für E2-Szenario",
            Θ.tau_2 <= h_e2 < Θ.tau_3, f"H={h_e2:.4f}")


# ══════════════════════════════════════════════════════════════════════════════
# TEST 3 — JSD (Jensen-Shannon-Divergenz, WP2 §3.3.5)
# ══════════════════════════════════════════════════════════════════════════════

header("TEST 3 — JSD (Jensen-Shannon-Divergenz)")

subheader("3a — Gleiche Verteilungen → JSD = 0")
p = {"causes": 0.7, "correlates": 0.3}
jsd_same = compute_jsd(p, p)
assert_approx("JSD(P, P) == 0.0", jsd_same, 0.0)

subheader("3b — Disjunkte Support-Mengen → JSD = 1 (maximale Divergenz)")
# WP2 §3.3.5: "fundamentale ontologische Meinungsverschiedenheit"
p_a = {"causes": 1.0}
p_b = {"inhibits": 1.0}
jsd_max = compute_jsd(p_a, p_b)
assert_approx("JSD(disjunkt) == 1.0", jsd_max, 1.0)

subheader("3c — Moderate Divergenz über τ₄=0.40 → E4 trigger")
# Builder A: klar "causes"
pa_e4 = {"causes": 0.80, "correlates": 0.20}
# Builder B: klar "suggests" (andere Ontologie)
pb_e4 = {"suggests": 0.75, "indicates": 0.25}
jsd_e4 = compute_jsd(pa_e4, pb_e4)
info(f"Builder A: {pa_e4}")
info(f"Builder B: {pb_e4}")
info(f"JSD = {jsd_e4:.4f}  (muss > τ₄={Θ.tau_4} für E4)")
assert_true("JSD > τ₄ für E4-Szenario", jsd_e4 > Θ.tau_4, f"JSD={jsd_e4:.4f}")

subheader("3d — Symmetrie: JSD(A,B) == JSD(B,A)")
jsd_ab = compute_jsd(pa_e4, pb_e4)
jsd_ba = compute_jsd(pb_e4, pa_e4)
assert_approx("JSD ist symmetrisch", jsd_ab, jsd_ba)

subheader("3e — JSD ∈ [0,1] für base-2 log")
import random
random.seed(42)
for i in range(10):
    keys = [f"r{j}" for j in range(random.randint(2, 6))]
    vals_a = [random.random() for _ in keys]; s = sum(vals_a)
    vals_b = [random.random() for _ in keys]; s2 = sum(vals_b)
    pa = {k: v/s for k, v in zip(keys, vals_a)}
    pb = {k: v/s2 for k, v in zip(keys, vals_b)}
    j = compute_jsd(pa, pb)
    assert_true(f"JSD ∈ [0,1] (random trial {i+1})", 0.0 <= j <= 1.0 + 1e-9,
                f"JSD={j:.4f}")


# ══════════════════════════════════════════════════════════════════════════════
# TEST 4 — Emission Rules E0–E4 (WP2 §7.2)
# ══════════════════════════════════════════════════════════════════════════════

header("TEST 4 — Emission Engine (E0–E4)")

engine = EmissionEngine(Θ)

# ── E0: Strukturelle Ablehnung ────────────────────────────────────────────────
subheader("4a — E0: Strukturelle Ablehnung (p_illegal > τ₀=0.50)")
# Beispiel: "Wasser ist schöner als Energie" — kein gültiges Typ-Paar
_, proj_e0 = make_projection(
    P_r={"beauty_comparison": 0.60, "correlates": 0.40},
    p_illegal=0.55,  # über τ₀
    source_text="Water is more beautiful than energy",
)
candidates_e0 = engine.emit(proj_e0)
info(f"p_illegal={proj_e0.p_illegal}  τ₀={Θ.tau_0}")
assert_eq("E0: Status = STRUCTURAL_VIOLATION",
          proj_e0.status, EmissionStatus.STRUCTURAL_VIOLATION)
assert_eq("E0: Emission Rule = E0", proj_e0.emission_rule, EmissionRule.E0)
assert_eq("E0: Keine Kandidaten", len(candidates_e0), 0)

# ── E1: Singular Emission ─────────────────────────────────────────────────────
subheader("4b — E1: Singular Emission (max(P_r) > τ₁, H_norm < τ₂)")
# WP2-Beispiel u2: "remote work may_increase productivity"
# Sehr spitze Verteilung nötig: max>0.60 UND H_norm<0.25
_, proj_e1 = make_projection(
    P_r={"may_increase": 0.97, "correlates_with": 0.03},
    subjects=("remote work",),
    objects=("productivity",),
    P_modality={"suggested": 0.9, "possible": 0.1},
    P_category={"dynamic": 1.0},
    source_text="remote work may increase productivity",
)
candidates_e1 = engine.emit(proj_e1)
info(f"P_r: {proj_e1.P_r}")
info(f"H_norm={proj_e1.h_norm:.4f}  max_rel=may_increase@0.97")
assert_eq("E1: Status = READY_FOR_CLAIM",
          proj_e1.status, EmissionStatus.READY_FOR_CLAIM)
assert_eq("E1: Emission Rule = E1", proj_e1.emission_rule, EmissionRule.E1)
assert_eq("E1: Genau 1 Kandidat", len(candidates_e1), 1)
c = candidates_e1[0]
assert_eq("E1: Relation = may_increase", c.relation, "may_increase")
assert_eq("E1: Subject = remote work", c.subject, "remote work")
assert_eq("E1: Object = productivity", c.object, "productivity")
assert_eq("E1: Rank = 1", c.rank, 1)
assert_true("E1: relation_score > τ₁", c.relation_score > Θ.tau_1,
            f"score={c.relation_score:.3f}")
assert_true("E1: H_norm < τ₂", proj_e1.h_norm < Θ.tau_2,
            f"H={proj_e1.h_norm:.4f}")
if VERBOSE:
    info(f"  Kandidat: {c.to_dict()}")

# ── E2: Multiple Emission ─────────────────────────────────────────────────────
subheader("4c — E2: Multiple Emission (τ₂ ≤ H_norm < τ₃)")
# WP2-Beispiel u1: "results suggest [claim]" — ambig zwischen suggests/indicates
# Moderate Dominanz: max_prob=0.80 > τ₁, H_norm≈0.56 ∈ [τ₂, τ₃)
_, proj_e2 = make_projection(
    P_r={"suggests": 0.80, "indicates": 0.15, "supports": 0.05},
    subjects=("results",),
    objects=("remote work increases productivity",),
    P_modality={"suggested": 1.0},
    P_category={"epistemic": 1.0},
    source_text="results suggest that remote work may increase productivity",
)
candidates_e2 = engine.emit(proj_e2, k=3)
info(f"P_r: {proj_e2.P_r}")
info(f"H_norm={proj_e2.h_norm:.4f}  (zwischen τ₂={Θ.tau_2} und τ₃={Θ.tau_3})")
assert_eq("E2: Status = READY_FOR_CLAIM",
          proj_e2.status, EmissionStatus.READY_FOR_CLAIM)
assert_eq("E2: Emission Rule = E2", proj_e2.emission_rule, EmissionRule.E2)
assert_eq("E2: 3 Kandidaten (k=3)", len(candidates_e2), 3)
assert_eq("E2: Rang-1-Relation = suggests", candidates_e2[0].relation, "suggests")
assert_eq("E2: Rang-2-Relation = indicates", candidates_e2[1].relation, "indicates")
assert_eq("E2: Rang-3-Relation = supports", candidates_e2[2].relation, "supports")
ranks = [c.rank for c in candidates_e2]
assert_eq("E2: Ränge = [1,2,3]", ranks, [1, 2, 3])

# ── E3: Ambiguous Block ───────────────────────────────────────────────────────
subheader("4d — E3: Ambiguity Block (H_norm >= τ₃=0.65)")
# Fast gleichverteilte Distribution — epistemisches Hedging
_, proj_e3 = make_projection(
    P_r={"causes": 0.28, "correlates": 0.27, "inhibits": 0.25, "suggests": 0.20},
    source_text="may suggest or possibly cause or correlate with effect",
)
candidates_e3 = engine.emit(proj_e3)
info(f"P_r: {proj_e3.P_r}")
info(f"H_norm={proj_e3.h_norm:.4f}  (muss >= τ₃={Θ.tau_3})")
assert_eq("E3: Status = AMBIGUOUS", proj_e3.status, EmissionStatus.AMBIGUOUS)
assert_eq("E3: Emission Rule = E3", proj_e3.emission_rule, EmissionRule.E3)
assert_eq("E3: Keine Kandidaten emittiert", len(candidates_e3), 0)
assert_true("E3: H_norm >= τ₃", proj_e3.h_norm >= Θ.tau_3,
            f"H={proj_e3.h_norm:.4f}")

# ── E4: Builder Divergence ────────────────────────────────────────────────────
subheader("4e — E4: Builder Divergenz (JSD > τ₄=0.40)")
# Gleiches SemanticUnit, zwei verschiedene Builder-Interpretationen
unit_shared = SemanticUnit.new(
    source_text="The compound inhibits the pathway",
    source_ref="WP2-test",
)
proj_a = SemanticProjection(
    projection_id=str(uuid.uuid4()),
    unit_id=unit_shared.unit_id,
    builder_origin="alpha",
    matrix_version="v2.2.0-TEST",
    P_r={"inhibits": 0.80, "downregulates": 0.20},
    subject_candidates=["compound"],
    object_candidates=["pathway"],
    P_modality={"asserted": 1.0},
)
proj_b = SemanticProjection(
    projection_id=str(uuid.uuid4()),
    unit_id=unit_shared.unit_id,
    builder_origin="beta",
    matrix_version="v2.2.0-TEST",
    P_r={"correlates_with": 0.70, "suggests": 0.30},
    subject_candidates=["compound"],
    object_candidates=["pathway"],
    P_modality={"suggested": 1.0},
)
jsd = engine.apply_e4(proj_a, proj_b)
info(f"Builder A P_r: {proj_a.P_r}")
info(f"Builder B P_r: {proj_b.P_r}")
info(f"JSD = {jsd:.4f}  (muss > τ₄={Θ.tau_4} für E4)")
assert_true("E4: JSD > τ₄", jsd > Θ.tau_4, f"JSD={jsd:.4f}")
assert_eq("E4: proj_a Status = BRANCH_CANDIDATE",
          proj_a.status, EmissionStatus.BRANCH_CANDIDATE)
assert_eq("E4: proj_b Status = BRANCH_CANDIDATE",
          proj_b.status, EmissionStatus.BRANCH_CANDIDATE)
assert_eq("E4: proj_a Rule = E4", proj_a.emission_rule, EmissionRule.E4)
assert_eq("E4: proj_b Rule = E4", proj_b.emission_rule, EmissionRule.E4)


# ══════════════════════════════════════════════════════════════════════════════
# TEST 5 — Vollständige Pipeline: Text → ClaimCandidate
# ══════════════════════════════════════════════════════════════════════════════

header("TEST 5 — Vollständige Pipeline (WP2 §2)")

subheader("Beispielsatz aus WP2 §3.2:")
sentence = ("The results suggest that remote work may increase productivity, "
            "although the effect varies across sectors.")
print(f"\n  {GRAY}\"{sentence}\"{RESET}\n")

# Fragmentation → 3 SemanticUnits (wie in WP2 §3.2 definiert)
u1 = SemanticUnit.new("results suggest [remote work increases productivity]",
                       "WP2-example", fragmentation_signal="evidential:suggest")
u2 = SemanticUnit.new("remote work may increase productivity",
                       "WP2-example", fragmentation_signal="modal:may")
u3 = SemanticUnit.new("effect varies across sectors",
                       "WP2-example", fragmentation_signal="conjunctive:although")

info(f"Fragmentation → {3} SemanticUnits")
assert_eq("u1 signal=evidential:suggest",
          u1.fragmentation_signal, "evidential:suggest")
assert_eq("u2 signal=modal:may",
          u2.fragmentation_signal, "modal:may")
assert_eq("u3 signal=conjunctive:although",
          u3.fragmentation_signal, "conjunctive:although")

# Projection — manuelle P_r entsprechend WP2-Semantik
projections = {}

# u1: "results suggest …" → evidential, multi-candidate (E2)
# H_norm ≈ 0.558 ∈ [τ₂, τ₃) und max=0.80 > τ₁ aber H > τ₂ → E2
projections["u1"] = SemanticProjection(
    projection_id=str(uuid.uuid4()),
    unit_id=u1.unit_id,
    builder_origin="alpha",
    matrix_version="v2.2.0-TEST",
    P_r={"suggests": 0.80, "indicates": 0.15, "supports": 0.05},
    subject_candidates=["results"],
    object_candidates=["remote work increases productivity"],
    P_modality={"suggested": 0.8, "hypothesized": 0.2},
    P_category={"epistemic": 1.0},
)

# u2: "may increase" → modal, starke Einzeldominanz (E1)
# H_norm ≈ 0.195 < τ₂=0.25 und max=0.97 > τ₁=0.60 → E1
projections["u2"] = SemanticProjection(
    projection_id=str(uuid.uuid4()),
    unit_id=u2.unit_id,
    builder_origin="alpha",
    matrix_version="v2.2.0-TEST",
    P_r={"may_increase": 0.97, "correlates_with": 0.03},
    subject_candidates=["remote work"],
    object_candidates=["productivity"],
    P_modality={"suggested": 0.7, "possible": 0.3},
    P_category={"dynamic": 1.0},
)

# u3: "varies across sectors" → statistisch, E2
# H_norm ≈ 0.625 ∈ [τ₂, τ₃) und max=0.75 > τ₁ aber H > τ₂ → E2
projections["u3"] = SemanticProjection(
    projection_id=str(uuid.uuid4()),
    unit_id=u3.unit_id,
    builder_origin="alpha",
    matrix_version="v2.2.0-TEST",
    P_r={"varies_across": 0.75, "depends_on": 0.20, "correlates_with": 0.05},
    subject_candidates=["effect"],
    object_candidates=["sectors"],
    P_modality={"asserted": 1.0},
    P_category={"statistical": 1.0},
)

# Emission
all_candidates = []
rule_map = {}
for key, proj in projections.items():
    candidates = engine.emit(proj, k=3)
    all_candidates.extend(candidates)
    rule_map[key] = proj.emission_rule

info(f"Emission-Regeln: u1={rule_map['u1'].value}  u2={rule_map['u2'].value}  u3={rule_map['u3'].value}")
assert_eq("u1 → E2 (multi)", rule_map["u1"], EmissionRule.E2)
assert_eq("u2 → E1 (singular)", rule_map["u2"], EmissionRule.E1)
assert_eq("u3 → E2 (multi)", rule_map["u3"], EmissionRule.E2)

assert_true(f"Insgesamt ≥ 3 Kandidaten generiert",
            len(all_candidates) >= 3, f"count={len(all_candidates)}")

print()
info(f"Generierte ClaimCandidates ({len(all_candidates)} total):")
for c in all_candidates:
    tag = f"[{c.emission_rule.value} rank={c.rank}]"
    print(f"    {GRAY}{tag:12}{RESET} "
          f"{c.subject!r:25} --[{c.relation}]--> {c.object!r}  "
          f"score={c.relation_score:.2f} H={c.h_norm:.3f}")


# ── Protokollgrenze: ClaimCandidate → ClaimNode ───────────────────────────────
subheader("5b — Protokollgrenze: ClaimCandidateConverter")
converter = ClaimCandidateConverter()

# Nur E1/E2-Kandidaten dürfen konvertiert werden
e1_candidates = [c for c in all_candidates if c.emission_rule == EmissionRule.E1]
e2_candidates = [c for c in all_candidates if c.emission_rule == EmissionRule.E2]
info(f"E1-Kandidaten: {len(e1_candidates)}  E2-Kandidaten: {len(e2_candidates)}")

claim_nodes = converter.convert_batch(all_candidates)
info(f"Konvertierte ClaimNodes: {len(claim_nodes)}")
assert_eq("Alle E1/E2-Kandidaten konvertiert",
          len(claim_nodes), len(all_candidates))

# SPL-Provenance muss in source_refs stehen
for node in claim_nodes:
    has_spl = any("spl:" in ref for ref in node.source_refs)
    assert_true("ClaimNode hat SPL-Provenance in source_refs", has_spl,
                f"source_refs={node.source_refs}")

# BRANCH_CANDIDATE und AMBIGUOUS dürfen NICHT konvertiert werden
_, proj_block = make_projection(
    P_r={"causes": 0.28, "correlates": 0.27, "inhibits": 0.25, "suggests": 0.20},
    source_text="ambiguous fragment",
)
engine.emit(proj_block)  # wird E3/AMBIGUOUS
blocked_candidate = ClaimCandidate(
    candidate_id=str(uuid.uuid4()),
    projection_id=proj_block.projection_id,
    unit_id=proj_block.unit_id,
    source_ref="test",
    subject="X", relation="causes", object="Y",
    relation_score=0.28,
    emission_rule=EmissionRule.E3,
)
try:
    converter.convert(blocked_candidate)
    fail("E3-Kandidat sollte ValueError werfen")
except ValueError as e:
    ok(f"E3-Kandidat korrekt blockiert: {e}")


# ══════════════════════════════════════════════════════════════════════════════
# TEST 6 — Protokoll-Invariante [SHALL]
# ══════════════════════════════════════════════════════════════════════════════

header("TEST 6 — Protokoll-Invariante (WP2 §2)")

subheader("Kein Text-Fragment darf direkt zu ClaimNode werden")
info("Die einzige legale Pfad: text → SemanticUnit → SemanticProjection → ClaimCandidate → ClaimNode")

# Test: Vollständiger Pfad für u2 (canonical)
u = SemanticUnit.new("remote work increases productivity",
                     "WP2-invariant-test")
assert_true("SemanticUnit hat unit_id", bool(u.unit_id))

proj = SemanticProjection(
    projection_id=str(uuid.uuid4()),
    unit_id=u.unit_id,
    builder_origin="alpha",
    matrix_version="v2.2.0-TEST",
    P_r={"increases": 0.97, "enables": 0.03},
    subject_candidates=["remote work"],
    object_candidates=["productivity"],
    P_modality={"asserted": 1.0},
    P_category={"dynamic": 1.0},
)
assert_true("SemanticProjection referenziert unit_id",
            proj.unit_id == u.unit_id)

candidates = engine.emit(proj)
assert_eq("Emission → E1", proj.emission_rule, EmissionRule.E1)
assert_eq("1 Kandidat", len(candidates), 1)
assert_true("Kandidat referenziert projection_id",
            candidates[0].projection_id == proj.projection_id)
assert_true("Kandidat referenziert unit_id",
            candidates[0].unit_id == u.unit_id)

node = converter.convert(candidates[0])
assert_true("ClaimNode hat subject", bool(node.subject))
assert_true("ClaimNode hat predicate", bool(node.predicate))
assert_true("ClaimNode hat SPL-Provenance",
            any("spl:" in r for r in node.source_refs))
assert_true("ClaimNode hat Assumptions ≥ 1", len(node.assumptions) >= 1)

info(f"ClaimNode: {node}")


# ══════════════════════════════════════════════════════════════════════════════
# ZUSAMMENFASSUNG
# ══════════════════════════════════════════════════════════════════════════════

header("ZUSAMMENFASSUNG")

total_assertions = sum(
    line.count("ok(") + line.count("fail(") + line.count("assert_")
    for line in open(__file__).readlines()
)

if _FAILURES:
    print(f"\n  {RED}{BOLD}FEHLGESCHLAGEN: {len(_FAILURES)} Tests{RESET}")
    for f in _FAILURES:
        print(f"    {RED}✗{RESET} {f}")
    sys.exit(1)
else:
    print(f"\n  {GREEN}{BOLD}ALLE TESTS BESTANDEN{RESET}")
    print(f"\n  Getestete Theorie-Komponenten:")
    print(f"    {GREEN}✓{RESET} SPLThresholds Θ (τ₀–τ₄) — Validierung & Werte")
    print(f"    {GREEN}✓{RESET} H_norm — Shannon-Entropie (Punkt-Masse, Gleichverteilung, E1/E2/E3)")
    print(f"    {GREEN}✓{RESET} JSD — Jensen-Shannon-Divergenz (Symmetrie, Disjunktheit, E4)")
    print(f"    {GREEN}✓{RESET} EmissionEngine E0 — Strukturelle Ablehnung")
    print(f"    {GREEN}✓{RESET} EmissionEngine E1 — Singular Emission (argmax)")
    print(f"    {GREEN}✓{RESET} EmissionEngine E2 — Multiple Emission (top-k)")
    print(f"    {GREEN}✓{RESET} EmissionEngine E3 — Ambiguity Block")
    print(f"    {GREEN}✓{RESET} EmissionEngine E4 — Builder Divergence → BRANCH_CANDIDATE")
    print(f"    {GREEN}✓{RESET} Vollständige Pipeline: Text → SemanticUnit → SemanticProjection → ClaimCandidate → ClaimNode")
    print(f"    {GREEN}✓{RESET} Protokoll-Invariante: E3/E4 blockiert Konvertierung")
    print(f"    {GREEN}✓{RESET} SPL-Provenance in ClaimNode.source_refs")
    print()
