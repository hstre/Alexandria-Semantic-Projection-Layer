"""
Alexandria SPL — Gateway Integration Test
==========================================
Verifiziert spl_gateway.py als protokoll-seitigen Einstiegspunkt.

Testet:
    1. SPLGateway Instanziierung (Default-Θ, custom Θ, ungültiges Θ)
    2. submit() — Einzel-Projektion für alle vier Emission-Pfade
    3. submit_dual() — Dual-Builder E4-Workflow
    4. submit_batch() — Batch-Verarbeitung
    5. to_claims() — Protokollgrenze (READY → ClaimNode)
    6. to_claims() Blocking (AMBIGUOUS, BRANCH_CANDIDATE)
    7. to_claims_batch() — gemischte Ergebnisse
    8. audit_log() — Vollständigkeit & Struktur
    9. summary() — Aggregatsstatistik
   10. make_gateway() — Fabrik-Funktion
   11. Validierungen — leeres P_r, P_r summe ≠ 1, unit_id-Mismatch

Ausführung:
    python test_gateway.py
    python test_gateway.py --verbose
"""

import sys
import types
import uuid

# ── Mock schema.py (wie in test_app.py) ──────────────────────────────────────
_mock_schema = types.ModuleType("spl_schema_mock")

class _Category:
    EMPIRICAL  = "EMPIRICAL"
    MODEL      = "MODEL"
    NORMATIVE  = "NORMATIVE"
    def __getitem__(self, item): return item

class _Modality:
    HYPOTHESIS = "hypothesis"
    def __init__(self, v): self.value = v

class _BuilderOrigin:
    ALPHA = "alpha"
    BETA  = "beta"

class _ClaimNode:
    def __init__(self, **kw):
        for k, v in kw.items(): setattr(self, k, v)
    @classmethod
    def new(cls, subject, predicate, object, category, assumptions, source_refs):
        return cls(subject=subject, predicate=predicate, object=object,
                   category=category, assumptions=assumptions,
                   source_refs=source_refs, modality=None, builder_origin=None)
    def __repr__(self):
        return (f"ClaimNode({self.subject!r} --[{self.predicate}]--> "
                f"{self.object!r})")

_mock_schema.ClaimNode       = _ClaimNode
_mock_schema.Category        = _Category()
_mock_schema.Modality        = _Modality
_mock_schema.BuilderOrigin   = _BuilderOrigin()
_mock_schema.EpistemicStatus = type("EpistemicStatus", (), {"UNVALIDATED": "unvalidated"})()

import spl as _spl_module
_spl_module.__package__ = "_spl_pkg"
sys.modules["_spl_pkg"]        = types.ModuleType("_spl_pkg")
sys.modules["_spl_pkg.schema"] = _mock_schema

# ── Imports ───────────────────────────────────────────────────────────────────

from spl import SemanticProjection, SemanticUnit, SPLThresholds, EmissionStatus, EmissionRule
from spl_gateway import (
    SPLGateway, SPLResult, DualBuilderResult,
    SPLGatewayError, make_gateway,
)

VERBOSE = "--verbose" in sys.argv or "-v" in sys.argv

# ── Farben & Assertions ───────────────────────────────────────────────────────

RESET = "\033[0m"; BOLD = "\033[1m"
GREEN = "\033[32m"; RED = "\033[31m"; CYAN = "\033[36m"; GRAY = "\033[90m"

_FAILURES = []

def ok(msg):   print(f"  {GREEN}✓{RESET} {msg}")
def fail(msg): print(f"  {RED}✗{RESET} {msg}"); _FAILURES.append(msg)
def info(msg): print(f"  {CYAN}→{RESET} {msg}")

def header(t):
    print(f"\n{BOLD}{'─'*60}{RESET}\n{BOLD}{t}{RESET}\n{BOLD}{'─'*60}{RESET}")

def subheader(t): print(f"\n  {CYAN}{t}{RESET}")

def assert_eq(label, got, expected):
    if got == expected: ok(f"{label}: {got!r}")
    else: fail(f"{label}: got {got!r}, expected {expected!r}")

def assert_true(label, cond, detail=""):
    if cond: ok(f"{label}{(' — ' + detail) if detail else ''}")
    else: fail(f"{label}{(' — ' + detail) if detail else ''} [FAILED]")

def assert_raises(label, exc_type, fn):
    try:
        fn()
        fail(f"{label}: expected {exc_type.__name__}, got no exception")
    except exc_type as e:
        ok(f"{label}: {exc_type.__name__} korrekt — {e}")
    except Exception as e:
        fail(f"{label}: falscher Exception-Typ {type(e).__name__}: {e}")


# ── Hilfsfunktionen ───────────────────────────────────────────────────────────

def _proj(P_r, subjects=("X",), objects=("Y",),
          builder="alpha", p_illegal=0.0,
          P_modality=None, P_category=None,
          unit_id=None) -> SemanticProjection:
    """Minimale SemanticProjection für Gateway-Tests."""
    return SemanticProjection(
        projection_id=str(uuid.uuid4()),
        unit_id=unit_id or str(uuid.uuid4()),
        builder_origin=builder,
        matrix_version="v2.2.0-TEST",
        P_r=P_r,
        subject_candidates=list(subjects),
        object_candidates=list(objects),
        P_modality=P_modality or {"asserted": 1.0},
        P_category=P_category or {"dynamic": 1.0},
        p_illegal=p_illegal,
    )

# Canonical P_r-Verteilungen (kalibriert auf Θ-Standard-Werte)
P_E1 = {"causes": 0.97, "correlates": 0.03}           # H≈0.19 < τ₂ → E1
P_E2 = {"suggests": 0.80, "indicates": 0.15, "supports": 0.05}  # H≈0.56 → E2
P_E3 = {"a": 0.26, "b": 0.25, "c": 0.25, "d": 0.24}  # H≈0.99 ≥ τ₃ → E3
P_E0_OK  = {"causes": 0.97, "correlates": 0.03}       # p_illegal=0   → ok
P_ILLEGAL = 0.55                                        # > τ₀=0.50 → E0


# ══════════════════════════════════════════════════════════════════════════════
# TEST 1 — Instanziierung
# ══════════════════════════════════════════════════════════════════════════════

header("TEST 1 — SPLGateway Instanziierung")

subheader("1a — Default-Θ")
gw = SPLGateway()
assert_eq("tau_1", gw.thresholds.tau_1, 0.60)
assert_eq("tau_3", gw.thresholds.tau_3, 0.65)
assert_true("audit_log startet leer", gw.audit_log() == [])

subheader("1b — Custom Θ via SPLGateway(thresholds=...)")
custom = SPLThresholds(tau_1=0.70, tau_2=0.20, tau_3=0.70)
gw_custom = SPLGateway(thresholds=custom)
assert_eq("custom tau_1", gw_custom.thresholds.tau_1, 0.70)

subheader("1c — make_gateway() Fabrik")
gw_factory = make_gateway(tau_1=0.75, tau_2=0.15, tau_3=0.70)
assert_eq("factory tau_1", gw_factory.thresholds.tau_1, 0.75)

subheader("1d — Ungültiges Θ → SPLGatewayError")
assert_raises("tau_2 > tau_3",
    SPLGatewayError,
    lambda: SPLGateway(thresholds=SPLThresholds(tau_2=0.80, tau_3=0.60))
)


# ══════════════════════════════════════════════════════════════════════════════
# TEST 2 — submit() Einzelprojektionen
# ══════════════════════════════════════════════════════════════════════════════

header("TEST 2 — submit() Einzelprojektionen")

gw = SPLGateway()

subheader("2a — E1: READY_FOR_CLAIM mit 1 Kandidat")
r_e1 = gw.submit(_proj(P_E1, subjects=("compound",), objects=("enzyme",)))
assert_true("Rückgabe ist SPLResult", isinstance(r_e1, SPLResult))
assert_eq("status=READY_FOR_CLAIM", r_e1.status, EmissionStatus.READY_FOR_CLAIM)
assert_eq("emission_rule=E1", r_e1.emission_rule, EmissionRule.E1)
assert_eq("1 Kandidat", len(r_e1.candidates), 1)
assert_true("is_ready()==True", r_e1.is_ready())
assert_true("is_blocked()==False", not r_e1.is_blocked())
assert_true("top_candidate() ist E1-Kandidat", r_e1.top_candidate() is not None)
assert_eq("top_candidate rank=1", r_e1.top_candidate().rank, 1)
assert_true("result_id ist UUID", len(r_e1.result_id) == 36)
if VERBOSE: info(f"to_dict: {r_e1.to_dict()}")

subheader("2b — E2: READY_FOR_CLAIM mit 3 Kandidaten")
r_e2 = gw.submit(_proj(P_E2), k=3)
assert_eq("status=READY_FOR_CLAIM", r_e2.status, EmissionStatus.READY_FOR_CLAIM)
assert_eq("emission_rule=E2", r_e2.emission_rule, EmissionRule.E2)
assert_eq("3 Kandidaten (k=3)", len(r_e2.candidates), 3)
assert_true("is_ready()==True", r_e2.is_ready())
ranks = [c.rank for c in r_e2.candidates]
assert_eq("Ränge [1,2,3]", ranks, [1, 2, 3])

subheader("2c — E3: AMBIGUOUS → keine Kandidaten")
r_e3 = gw.submit(_proj(P_E3))
assert_eq("status=AMBIGUOUS", r_e3.status, EmissionStatus.AMBIGUOUS)
assert_eq("emission_rule=E3", r_e3.emission_rule, EmissionRule.E3)
assert_eq("0 Kandidaten", len(r_e3.candidates), 0)
assert_true("is_blocked()==True", r_e3.is_blocked())
assert_true("is_ready()==False", not r_e3.is_ready())
assert_true("top_candidate()==None", r_e3.top_candidate() is None)

subheader("2d — E0: STRUCTURAL_VIOLATION → keine Kandidaten")
r_e0 = gw.submit(_proj(P_E0_OK, p_illegal=P_ILLEGAL))
assert_eq("status=STRUCTURAL_VIOLATION",
          r_e0.status, EmissionStatus.STRUCTURAL_VIOLATION)
assert_eq("emission_rule=E0", r_e0.emission_rule, EmissionRule.E0)
assert_eq("0 Kandidaten", len(r_e0.candidates), 0)
assert_true("is_blocked()==True", r_e0.is_blocked())


# ══════════════════════════════════════════════════════════════════════════════
# TEST 3 — submit_dual() — E4-Workflow
# ══════════════════════════════════════════════════════════════════════════════

header("TEST 3 — submit_dual() (E4 Builder-Divergenz)")

gw = SPLGateway()
shared_uid = str(uuid.uuid4())

subheader("3a — Disjunkte Ontologien → JSD=1.0 → branched=True")
pa = _proj({"inhibits": 1.0}, unit_id=shared_uid, builder="alpha")
pb = _proj({"enables":  1.0}, unit_id=shared_uid, builder="beta")
dual_branch = gw.submit_dual(pa, pb)
assert_true("Rückgabe ist DualBuilderResult", isinstance(dual_branch, DualBuilderResult))
assert_true("JSD=1.0", abs(dual_branch.jsd - 1.0) < 1e-9, f"JSD={dual_branch.jsd}")
assert_true("branched=True", dual_branch.branched)
assert_eq("unit_id korrekt", dual_branch.unit_id, shared_uid)
assert_eq("alpha → BRANCH_CANDIDATE",
          dual_branch.result_alpha.status, EmissionStatus.BRANCH_CANDIDATE)
assert_eq("beta → BRANCH_CANDIDATE",
          dual_branch.result_beta.status, EmissionStatus.BRANCH_CANDIDATE)
assert_true("dual_id ist UUID", len(dual_branch.dual_id) == 36)
if VERBOSE: info(f"DualBuilderResult: jsd={dual_branch.jsd:.4f} branched={dual_branch.branched}")

subheader("3b — Gleiche Ontologie → JSD=0 → branched=False")
shared_uid2 = str(uuid.uuid4())
pc = _proj(P_E1, unit_id=shared_uid2, builder="alpha")
pd = _proj(P_E1, unit_id=shared_uid2, builder="beta")
dual_ok = gw.submit_dual(pc, pd)
assert_true("JSD≈0", dual_ok.jsd < 1e-9, f"JSD={dual_ok.jsd}")
assert_true("branched=False", not dual_ok.branched)
assert_eq("alpha → READY_FOR_CLAIM",
          dual_ok.result_alpha.status, EmissionStatus.READY_FOR_CLAIM)

subheader("3c — unit_id Mismatch → SPLGatewayError")
pe = _proj(P_E1, unit_id=str(uuid.uuid4()), builder="alpha")
pf = _proj(P_E1, unit_id=str(uuid.uuid4()), builder="beta")
assert_raises("unit_id-Mismatch", SPLGatewayError, lambda: gw.submit_dual(pe, pf))


# ══════════════════════════════════════════════════════════════════════════════
# TEST 4 — submit_batch()
# ══════════════════════════════════════════════════════════════════════════════

header("TEST 4 — submit_batch()")

gw = SPLGateway()
projs = [
    _proj(P_E1),  # → E1
    _proj(P_E2),  # → E2
    _proj(P_E3),  # → E3
    _proj(P_E1, p_illegal=P_ILLEGAL),  # → E0
    _proj(P_E1),  # → E1
]
results = gw.submit_batch(projs, k=3)
assert_eq("5 Ergebnisse", len(results), 5)
statuses = [r.status for r in results]
assert_eq("Reihenfolge erhalten [E1,E2,E3,E0,E1]",
          [s.value for s in statuses],
          ["ready_for_claim", "ready_for_claim", "ambiguous",
           "structural_violation", "ready_for_claim"])

ready = [r for r in results if r.is_ready()]
assert_eq("3 READY-Ergebnisse", len(ready), 3)


# ══════════════════════════════════════════════════════════════════════════════
# TEST 5 — to_claims() Protokollgrenze
# ══════════════════════════════════════════════════════════════════════════════

header("TEST 5 — to_claims() Protokollgrenze")

gw = SPLGateway()

subheader("5a — READY (E1) → ClaimNode erzeugt")
r = gw.submit(_proj(P_E1, subjects=("remote work",), objects=("productivity",),
                    P_category={"dynamic": 1.0}, P_modality={"asserted": 1.0}))
claims = gw.to_claims(r)
assert_eq("1 ClaimNode", len(claims), 1)
node = claims[0]
assert_eq("subject=remote work", node.subject, "remote work")
assert_eq("predicate=causes", node.predicate, "causes")
assert_true("SPL-Provenance in source_refs",
            any("spl:" in ref for ref in node.source_refs),
            f"refs={node.source_refs}")
assert_true("Assumptions nicht leer", len(node.assumptions) >= 1)

subheader("5b — READY (E2) → mehrere ClaimNodes")
r2 = gw.submit(_proj(P_E2), k=3)
claims2 = gw.to_claims(r2, extra_assumptions=["from domain: biology"])
assert_eq("3 ClaimNodes (k=3)", len(claims2), 3)
for cn in claims2:
    has_spl = any("spl:" in ref for ref in cn.source_refs)
    assert_true("SPL-Provenance vorhanden", has_spl)
    has_extra = any("biology" in a for a in cn.assumptions)
    assert_true("extra_assumption weitergegeben", has_extra)

subheader("5c — AMBIGUOUS → SPLGatewayError")
r_block = gw.submit(_proj(P_E3))
assert_raises("AMBIGUOUS → Fehler", SPLGatewayError, lambda: gw.to_claims(r_block))

subheader("5d — STRUCTURAL_VIOLATION → SPLGatewayError")
r_e0 = gw.submit(_proj(P_E0_OK, p_illegal=P_ILLEGAL))
assert_raises("E0 → Fehler", SPLGatewayError, lambda: gw.to_claims(r_e0))

subheader("5e — BRANCH_CANDIDATE → SPLGatewayError")
uid = str(uuid.uuid4())
pa = _proj({"inhibits": 1.0}, unit_id=uid)
pb = _proj({"enables":  1.0}, unit_id=uid)
dual = gw.submit_dual(pa, pb)
assert_raises("BRANCH_CANDIDATE → Fehler",
              SPLGatewayError, lambda: gw.to_claims(dual.result_alpha))


# ══════════════════════════════════════════════════════════════════════════════
# TEST 6 — to_claims_batch() gemischte Ergebnisse
# ══════════════════════════════════════════════════════════════════════════════

header("TEST 6 — to_claims_batch() gemischte Ergebnisse")

gw = SPLGateway()
mixed = gw.submit_batch([
    _proj(P_E1),   # ready → 1 claim
    _proj(P_E3),   # blocked
    _proj(P_E2),   # ready → 3 claims
    _proj(P_E0_OK, p_illegal=P_ILLEGAL),  # blocked
    _proj(P_E1),   # ready → 1 claim
])
all_claims = gw.to_claims_batch(mixed)
assert_eq("Insgesamt 5 ClaimNodes (1+3+1)", len(all_claims), 5)

skips = [e for e in gw.audit_log() if e["event"] == "to_claims_batch_skip"]
assert_eq("2 Skips geloggt", len(skips), 2)
skip_statuses = {s["status"] for s in skips}
assert_true("Ambiguous und E0 geloggt",
            "ambiguous" in skip_statuses and "structural_violation" in skip_statuses,
            f"statuses={skip_statuses}")


# ══════════════════════════════════════════════════════════════════════════════
# TEST 7 — Validierungen
# ══════════════════════════════════════════════════════════════════════════════

header("TEST 7 — Eingabe-Validierungen")

gw = SPLGateway()

subheader("7a — Leeres P_r → SPLGatewayError")
empty_proj = _proj({})
assert_raises("leeres P_r", SPLGatewayError, lambda: gw.submit(empty_proj))

subheader("7b — P_r Summe ≠ 1 → SPLGatewayError")
bad_sum = _proj({"causes": 0.50, "correlates": 0.10})  # Summe=0.60
assert_raises("P_r Summe ≠ 1", SPLGatewayError, lambda: gw.submit(bad_sum))

subheader("7c — P_r Summe ≈ 1 (Rundungsfehler ≤ 0.01 toleriert)")
almost = _proj({"causes": 0.9999, "correlates": 0.0002})  # Summe=1.0001
r = gw.submit(almost)
assert_true("Runde P_r wird akzeptiert", r is not None)


# ══════════════════════════════════════════════════════════════════════════════
# TEST 8 — audit_log() Vollständigkeit
# ══════════════════════════════════════════════════════════════════════════════

header("TEST 8 — audit_log() Vollständigkeit")

gw = SPLGateway()
r1 = gw.submit(_proj(P_E1))
r2 = gw.submit(_proj(P_E3))
gw.to_claims(r1)

log = gw.audit_log()
events = [e["event"] for e in log]
assert_eq("3 Log-Einträge", len(log), 3)
assert_eq("Events: [submit, submit, to_claims]", events,
          ["submit", "submit", "to_claims"])

submit_entry = log[0]
required_fields = {"event", "result_id", "unit_id", "projection_id",
                   "status", "emission_rule", "h_norm",
                   "candidate_count", "builder_origin", "matrix_version", "timestamp"}
missing = required_fields - set(submit_entry.keys())
assert_true("Alle Pflichtfelder im submit-Eintrag", len(missing) == 0,
            f"fehlend: {missing}")

claim_entry = log[2]
assert_eq("to_claims-Eintrag hat claim_count=1",
          claim_entry["claim_count"], 1)

subheader("8b — submit_dual loggt dual-Event")
gw2 = SPLGateway()
uid = str(uuid.uuid4())
gw2.submit_dual(_proj(P_E1, unit_id=uid), _proj(P_E1, unit_id=uid))
dual_events = [e for e in gw2.audit_log() if e["event"] == "submit_dual"]
assert_eq("1 submit_dual-Event", len(dual_events), 1)
assert_true("dual_id im Log", "dual_id" in dual_events[0])
assert_true("jsd im Log", "jsd" in dual_events[0])
assert_true("tau_4 im Log", "tau_4" in dual_events[0])


# ══════════════════════════════════════════════════════════════════════════════
# TEST 9 — summary()
# ══════════════════════════════════════════════════════════════════════════════

header("TEST 9 — summary()")

gw = SPLGateway()
results_for_summary = gw.submit_batch([
    _proj(P_E1),                        # E1 ready
    _proj(P_E2),                        # E2 ready
    _proj(P_E3),                        # E3 ambiguous
    _proj(P_E0_OK, p_illegal=P_ILLEGAL), # E0 violation
])
gw.to_claims(results_for_summary[0])   # 1 claim
gw.to_claims(results_for_summary[1])   # 3 claims

s = gw.summary()
info(f"Summary: {s}")
assert_eq("4 Submissions", s["submissions"], 4)
assert_eq("2 READY", s["by_status"].get("ready_for_claim", 0), 2)
assert_eq("1 AMBIGUOUS", s["by_status"].get("ambiguous", 0), 1)
assert_eq("1 STRUCTURAL_VIOLATION", s["by_status"].get("structural_violation", 0), 1)
assert_eq("E1: 1", s["by_emission_rule"].get("E1", 0), 1)
assert_eq("E2: 1", s["by_emission_rule"].get("E2", 0), 1)
assert_eq("4 Kandidaten total (1 E1 + 3 E2)", s["total_candidates"], 4)
assert_eq("4 ClaimNodes total (1+3)", s["total_claims"], 4)
assert_true("Thresholds in summary", "tau_1" in s["thresholds"])


# ══════════════════════════════════════════════════════════════════════════════
# TEST 10 — Vollständiger Protokoll-Workflow (WP2 §2)
# ══════════════════════════════════════════════════════════════════════════════

header("TEST 10 — Vollständiger Protokoll-Workflow")

print(f"\n  {GRAY}Szenario: Protokoll verarbeitet ein Dokument mit 4 SemanticUnits{RESET}")
print(f"  {GRAY}davon: 1×E1, 1×E2, 1×E3 (blockiert), 1×dual-builder (E4 → branch){RESET}\n")

gateway = SPLGateway()
uid_dual = str(uuid.uuid4())

document_projections = [
    _proj(P_E1, subjects=("CRISPR",), objects=("gene expression",),
          P_category={"dynamic": 1.0}, P_modality={"asserted": 1.0}),
    _proj(P_E2, subjects=("diet",), objects=("cardiovascular risk",),
          P_category={"statistical": 1.0}, P_modality={"suggested": 1.0}),
    _proj(P_E3, subjects=("factor X",), objects=("outcome Y",)),  # ambiguous
]
alpha_dual = _proj({"inhibits": 1.0}, subjects=("drug A",),
                   objects=("receptor B",), unit_id=uid_dual, builder="alpha")
beta_dual  = _proj({"activates": 1.0}, subjects=("drug A",),
                   objects=("receptor B",), unit_id=uid_dual, builder="beta")

# Batch submit
results = gateway.submit_batch(document_projections)

# Dual submit
dual = gateway.submit_dual(alpha_dual, beta_dual)

# Konvertierung
ready = [r for r in results if r.is_ready()]
claims = gateway.to_claims_batch(ready)

# Auswertung
s = gateway.summary()
info(f"Submission-Status: {s['by_status']}")
info(f"Emission-Regeln:   {s['by_emission_rule']}")
info(f"ClaimNodes:        {s['total_claims']}")
info(f"Dual branched:     {dual.branched}  (JSD={dual.jsd:.4f})")

assert_eq("2 READY aus Batch", len(ready), 2)
assert_eq("4 ClaimNodes (1+3)", len(claims), 4)
assert_true("Dual branched", dual.branched)

print()
info("Generierte ClaimNodes:")
for c in claims:
    print(f"    {GRAY}{c.subject!r:20} --[{c.predicate}]--> {c.object!r}{RESET}")

assert_true("Protokoll-Workflow abgeschlossen", True)


# ══════════════════════════════════════════════════════════════════════════════
# ZUSAMMENFASSUNG
# ══════════════════════════════════════════════════════════════════════════════

header("ZUSAMMENFASSUNG")

if _FAILURES:
    print(f"\n  {RED}{BOLD}FEHLGESCHLAGEN: {len(_FAILURES)} Tests{RESET}")
    for f in _FAILURES:
        print(f"    {RED}✗{RESET} {f}")
    sys.exit(1)
else:
    print(f"\n  {GREEN}{BOLD}ALLE GATEWAY-TESTS BESTANDEN{RESET}")
    print(f"\n  Getestete Protokoll-Schnittstellen:")
    print(f"    {GREEN}✓{RESET} SPLGateway Instanziierung (Default, Custom, Fabrik, Fehler)")
    print(f"    {GREEN}✓{RESET} submit() — E0/E1/E2/E3 Einzelprojektionen")
    print(f"    {GREEN}✓{RESET} submit_dual() — E4 Dual-Builder (branch + kein branch + Mismatch)")
    print(f"    {GREEN}✓{RESET} submit_batch() — Batch-Verarbeitung mit Reihenfolge-Garantie")
    print(f"    {GREEN}✓{RESET} to_claims() — READY_FOR_CLAIM → ClaimNode (Protokollgrenze)")
    print(f"    {GREEN}✓{RESET} to_claims() — Blocking von E0/E3/E4 (SPLGatewayError)")
    print(f"    {GREEN}✓{RESET} to_claims_batch() — gemischte Ergebnisse + Skip-Log")
    print(f"    {GREEN}✓{RESET} Eingabe-Validierung (leeres P_r, Summe ≠ 1, unit_id-Mismatch)")
    print(f"    {GREEN}✓{RESET} audit_log() — vollständige Ereignisstruktur")
    print(f"    {GREEN}✓{RESET} summary() — Aggregationsstatistik")
    print(f"    {GREEN}✓{RESET} Vollständiger Protokoll-Workflow (Batch + Dual + Konvertierung)")
    print()
