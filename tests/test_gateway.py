"""
tests/test_gateway.py
=====================
Unit tests for SPLGateway — the protocol-callable interface layer.

Tests:
    - emit_claim_nodes(): single entry point, validation, claim_id hash
    - _validate_candidate(): all 5 validation criteria
    - validate_claim_node(): structural completeness check
    - canonicalize_text(), canonicalize_entities(), hash_claim()
    - GatewayEvent logging and audit_log.json persistence
    - submit(), submit_dual(), submit_batch()
    - to_claims() / to_claims_batch() (legacy aliases)
    - emit_claims_from_results() batch API
    - summary() aggregation
    - CandidateRejectedError, ClaimValidationError, SPLGatewayError
    - Determinism: same triple → same claim_id

Run:
    python -m unittest tests/test_gateway.py
"""

import sys
import os
import json
import tempfile
import unittest
import uuid

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import tests._mock_schema as _mock
_mock.setup()

from spl import SemanticProjection, EmissionRule, EmissionStatus, SPLThresholds
from spl_gateway import (
    SPLGateway, SPLResult, DualBuilderResult, GatewayEvent,
    SPLGatewayError, CandidateRejectedError, ClaimValidationError,
    canonicalize_text, canonicalize_entities, hash_claim,
    validate_claim_node, make_gateway, MIN_EVIDENCE,
)

Θ = SPLThresholds()

# Calibrated P_r distributions (see test_spl_rules.py for rationale)
P_E1 = {"causes": 0.97, "correlates": 0.03}           # H≈0.19 < τ₂ → E1
P_E2 = {"suggests": 0.80, "indicates": 0.15, "supports": 0.05}  # H≈0.56 → E2
P_E3 = {"a": 0.28, "b": 0.27, "c": 0.25, "d": 0.20}  # H≈0.99 → E3
P_E0_ILLEGAL = 0.55                                     # > τ₀ → E0


def _proj(P_r, subjects=("X",), objects=("Y",), builder="alpha",
          p_illegal=0.0, unit_id=None,
          P_modality=None, P_category=None):
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


class TestUtilityFunctions(unittest.TestCase):
    """canonicalize_text, canonicalize_entities, hash_claim, validate_claim_node."""

    def test_canonicalize_text_lowercase(self):
        self.assertEqual(canonicalize_text("Paris"), "paris")

    def test_canonicalize_text_whitespace(self):
        self.assertEqual(canonicalize_text("  remote   work  "), "remote work")

    def test_canonicalize_text_strip(self):
        self.assertEqual(canonicalize_text("  hello  "), "hello")

    def test_canonicalize_entities_removes_parens(self):
        self.assertEqual(canonicalize_entities("Paris (city)"), "paris")

    def test_hash_claim_deterministic(self):
        h1 = hash_claim("Paris", "capital_of", "France")
        h2 = hash_claim("Paris", "capital_of", "France")
        self.assertEqual(h1, h2)

    def test_hash_claim_length(self):
        """SHA256 hex digest is always 64 characters."""
        h = hash_claim("A", "B", "C")
        self.assertEqual(len(h), 64)

    def test_hash_claim_case_insensitive(self):
        """Canonicalization makes hashing case-insensitive."""
        h1 = hash_claim("paris", "capital_of", "france")
        h2 = hash_claim("Paris", "capital_of", "France")
        self.assertEqual(h1, h2)

    def test_hash_claim_different_triples(self):
        """Different triples must produce different hashes."""
        h1 = hash_claim("Paris", "capital_of", "France")
        h2 = hash_claim("Paris", "capital_of", "Germany")
        self.assertNotEqual(h1, h2)

    def test_validate_claim_node_valid(self):
        """Valid node must not raise."""
        from tests._mock_schema import setup
        node = type("Node", (), {
            "subject": "Paris",
            "predicate": "capital_of",
            "object": "France",
            "source_refs": ["spl:abc/def/EE1"],
        })()
        validate_claim_node(node)  # must not raise

    def test_validate_claim_node_missing_subject(self):
        node = type("Node", (), {
            "subject": "",
            "predicate": "capital_of",
            "object": "France",
            "source_refs": ["spl:x"],
        })()
        with self.assertRaises(ClaimValidationError):
            validate_claim_node(node)

    def test_validate_claim_node_missing_source_refs(self):
        node = type("Node", (), {
            "subject": "Paris",
            "predicate": "capital_of",
            "object": "France",
            "source_refs": [],
        })()
        with self.assertRaises(ClaimValidationError):
            validate_claim_node(node)

    def test_validate_claim_node_none_source_refs(self):
        node = type("Node", (), {
            "subject": "A",
            "predicate": "B",
            "object": "C",
            "source_refs": None,
        })()
        with self.assertRaises(ClaimValidationError):
            validate_claim_node(node)


class TestEmitClaimNodes(unittest.TestCase):
    """emit_claim_nodes() — the single entry point."""

    def _gateway(self):
        return SPLGateway(audit_log_path=None)  # no file I/O in tests

    def test_emit_valid_candidate(self):
        gw = self._gateway()
        result = gw.submit(_proj(P_E1, subjects=("Paris",), objects=("France",)))
        self.assertTrue(result.is_ready())
        nodes = gw.emit_claim_nodes(result.candidates)
        self.assertEqual(len(nodes), 1)
        self.assertEqual(nodes[0].subject, "Paris")
        self.assertEqual(nodes[0].predicate, "causes")

    def test_emit_assigns_claim_id(self):
        gw = self._gateway()
        result = gw.submit(_proj(P_E1))
        nodes = gw.emit_claim_nodes(result.candidates)
        self.assertTrue(hasattr(nodes[0], "claim_id"))
        self.assertEqual(len(nodes[0].claim_id), 64)

    def test_emit_claim_id_is_deterministic(self):
        gw = self._gateway()
        r1 = gw.submit(_proj(P_E1, subjects=("Paris",), objects=("France",)))
        r2 = gw.submit(_proj(P_E1, subjects=("Paris",), objects=("France",)))
        n1 = gw.emit_claim_nodes(r1.candidates)
        n2 = gw.emit_claim_nodes(r2.candidates)
        self.assertEqual(n1[0].claim_id, n2[0].claim_id)

    def test_emit_e2_multiple_nodes(self):
        gw = self._gateway()
        result = gw.submit(_proj(P_E2), k=3)
        nodes = gw.emit_claim_nodes(result.candidates)
        self.assertEqual(len(nodes), 3)

    def test_emit_returns_empty_for_empty_candidates(self):
        gw = self._gateway()
        nodes = gw.emit_claim_nodes([])
        self.assertEqual(nodes, [])

    def test_emit_logs_gateway_events(self):
        gw = self._gateway()
        result = gw.submit(_proj(P_E1))
        gw.emit_claim_nodes(result.candidates)
        events = [e for e in gw.audit_log() if e.get("event") == "gateway_event"]
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]["decision"], "EMITTED")

    def test_emit_reject_high_entropy_e1(self):
        """E1 candidate with h_norm >= τ₂ must be rejected."""
        gw = self._gateway()
        # Build E2 result, then forge candidate as E1 with high entropy
        result = gw.submit(_proj(P_E2))
        # Patch emission_rule to E1 to force validation
        for c in result.candidates:
            c.emission_rule = EmissionRule.E1
            # h_norm inherited from E2 projection (≈ 0.56 > τ₂=0.25)
        nodes = gw.emit_claim_nodes(result.candidates)
        # All should be rejected (h_norm >= τ₂ for E1 check)
        self.assertEqual(nodes, [])
        events = [e for e in gw.audit_log()
                  if e.get("event") == "gateway_event" and e["decision"] == "REJECTED"]
        self.assertEqual(len(events), len(result.candidates))

    def test_emit_reject_low_confidence_e1(self):
        """E1 candidate with relation_score < τ₁ must be rejected."""
        gw = self._gateway()
        result = gw.submit(_proj(P_E1))
        # Tamper with score
        result.candidates[0].relation_score = 0.30  # < τ₁=0.60
        nodes = gw.emit_claim_nodes(result.candidates)
        self.assertEqual(nodes, [])

    def test_emit_reject_jsd_too_high(self):
        """JSD > τ₄ passed to emit_claim_nodes must reject all candidates."""
        gw = self._gateway()
        result = gw.submit(_proj(P_E1))
        # Simulate: jsd=0.99 > τ₄=0.40
        nodes = gw.emit_claim_nodes(result.candidates, jsd=0.99)
        self.assertEqual(nodes, [])

    def test_emit_reject_insufficient_evidence(self):
        """evidence_count < MIN_EVIDENCE must reject all candidates."""
        gw = self._gateway()
        result = gw.submit(_proj(P_E1))
        nodes = gw.emit_claim_nodes(result.candidates, evidence_count=0)
        self.assertEqual(nodes, [])

    def test_emit_jsd_below_tau4_passes(self):
        """JSD ≤ τ₄ must not reject candidates."""
        gw = self._gateway()
        result = gw.submit(_proj(P_E1))
        nodes = gw.emit_claim_nodes(result.candidates, jsd=0.10)
        self.assertEqual(len(nodes), 1)


class TestGatewayValidation(unittest.TestCase):
    """SPLGateway input validation."""

    def _gateway(self):
        return SPLGateway(audit_log_path=None)

    def test_empty_P_r_raises(self):
        gw = self._gateway()
        with self.assertRaises(SPLGatewayError):
            gw.submit(_proj({}))

    def test_P_r_sum_not_one_raises(self):
        gw = self._gateway()
        with self.assertRaises(SPLGatewayError):
            gw.submit(_proj({"a": 0.40, "b": 0.20}))

    def test_P_r_rounding_tolerance(self):
        """Sum within 0.01 of 1.0 is accepted."""
        gw = self._gateway()
        result = gw.submit(_proj({"a": 0.9999, "b": 0.0002}))
        self.assertIsNotNone(result)

    def test_invalid_thresholds_raise_on_init(self):
        with self.assertRaises(SPLGatewayError):
            SPLGateway(thresholds=SPLThresholds(tau_2=0.80, tau_3=0.60))

    def test_submit_dual_unit_id_mismatch_raises(self):
        gw = self._gateway()
        pa = _proj(P_E1, unit_id=str(uuid.uuid4()))
        pb = _proj(P_E1, unit_id=str(uuid.uuid4()))
        with self.assertRaises(SPLGatewayError):
            gw.submit_dual(pa, pb)


class TestSubmitAndResults(unittest.TestCase):
    """submit(), submit_dual(), submit_batch() — result structure."""

    def _gateway(self):
        return SPLGateway(audit_log_path=None)

    def test_submit_e1_result(self):
        gw = self._gateway()
        r = gw.submit(_proj(P_E1))
        self.assertIsInstance(r, SPLResult)
        self.assertEqual(r.status, EmissionStatus.READY_FOR_CLAIM)
        self.assertEqual(r.emission_rule, EmissionRule.E1)
        self.assertEqual(len(r.candidates), 1)
        self.assertTrue(r.is_ready())

    def test_submit_e3_result_blocked(self):
        gw = self._gateway()
        r = gw.submit(_proj(P_E3))
        self.assertEqual(r.status, EmissionStatus.AMBIGUOUS)
        self.assertTrue(r.is_blocked())
        self.assertEqual(r.candidates, [])

    def test_submit_dual_branch(self):
        gw = self._gateway()
        uid = str(uuid.uuid4())
        pa = _proj({"inhibits": 1.0}, unit_id=uid, builder="alpha")
        pb = _proj({"enables": 1.0},  unit_id=uid, builder="beta")
        dual = gw.submit_dual(pa, pb)
        self.assertIsInstance(dual, DualBuilderResult)
        self.assertTrue(dual.branched)
        self.assertAlmostEqual(dual.jsd, 1.0)
        self.assertEqual(dual.result_alpha.status, EmissionStatus.BRANCH_CANDIDATE)
        self.assertEqual(dual.result_beta.status, EmissionStatus.BRANCH_CANDIDATE)

    def test_submit_dual_no_branch(self):
        gw = self._gateway()
        uid = str(uuid.uuid4())
        pa = _proj(P_E1, unit_id=uid, builder="alpha")
        pb = _proj(P_E1, unit_id=uid, builder="beta")
        dual = gw.submit_dual(pa, pb)
        self.assertFalse(dual.branched)
        self.assertAlmostEqual(dual.jsd, 0.0)
        self.assertTrue(dual.result_alpha.is_ready())

    def test_submit_batch_order_preserved(self):
        gw = self._gateway()
        projs = [_proj(P_E1), _proj(P_E3), _proj(P_E2)]
        results = gw.submit_batch(projs)
        self.assertEqual(len(results), 3)
        self.assertEqual(results[0].emission_rule, EmissionRule.E1)
        self.assertEqual(results[1].emission_rule, EmissionRule.E3)
        self.assertEqual(results[2].emission_rule, EmissionRule.E2)


class TestToClaimsLegacy(unittest.TestCase):
    """to_claims() / to_claims_batch() — backward-compatible aliases."""

    def _gateway(self):
        return SPLGateway(audit_log_path=None)

    def test_to_claims_ready_result(self):
        gw = self._gateway()
        r = gw.submit(_proj(P_E1))
        claims = gw.to_claims(r)
        self.assertEqual(len(claims), 1)

    def test_to_claims_blocked_raises(self):
        gw = self._gateway()
        r = gw.submit(_proj(P_E3))
        with self.assertRaises(SPLGatewayError):
            gw.to_claims(r)

    def test_to_claims_branch_raises(self):
        gw = self._gateway()
        uid = str(uuid.uuid4())
        dual = gw.submit_dual(
            _proj({"inhibits": 1.0}, unit_id=uid),
            _proj({"enables": 1.0},  unit_id=uid),
        )
        with self.assertRaises(SPLGatewayError):
            gw.to_claims(dual.result_alpha)

    def test_to_claims_batch_mixed(self):
        gw = self._gateway()
        results = gw.submit_batch([_proj(P_E1), _proj(P_E3), _proj(P_E2)])
        claims = gw.to_claims_batch(results)
        # E1 → 1, E3 → 0, E2 → 3 (k=3 default)
        self.assertEqual(len(claims), 4)

    def test_to_claims_batch_skip_logged(self):
        gw = self._gateway()
        results = gw.submit_batch([_proj(P_E1), _proj(P_E3)])
        gw.to_claims_batch(results)
        skips = [e for e in gw.audit_log()
                 if e.get("event") == "to_claims_batch_skip"]
        self.assertEqual(len(skips), 1)
        self.assertEqual(skips[0]["status"], "ambiguous")


class TestAuditLog(unittest.TestCase):
    """audit_log() and GatewayEvent persistence."""

    def test_audit_log_initially_empty(self):
        gw = SPLGateway(audit_log_path=None)
        self.assertEqual(gw.audit_log(), [])

    def test_submit_appends_to_log(self):
        gw = SPLGateway(audit_log_path=None)
        gw.submit(_proj(P_E1))
        log = gw.audit_log()
        submit_events = [e for e in log if e.get("event") == "submit"]
        self.assertEqual(len(submit_events), 1)

    def test_submit_log_entry_fields(self):
        gw = SPLGateway(audit_log_path=None)
        r = gw.submit(_proj(P_E1))
        entry = [e for e in gw.audit_log() if e.get("event") == "submit"][0]
        for field in ("result_id", "unit_id", "projection_id", "status",
                      "emission_rule", "h_norm", "candidate_count",
                      "builder_origin", "matrix_version", "timestamp"):
            self.assertIn(field, entry, f"Missing field: {field}")

    def test_emit_claim_nodes_logs_gateway_event(self):
        gw = SPLGateway(audit_log_path=None)
        r = gw.submit(_proj(P_E1))
        gw.emit_claim_nodes(r.candidates)
        events = [e for e in gw.audit_log() if e.get("event") == "gateway_event"]
        self.assertEqual(len(events), 1)
        self.assertIn("decision", events[0])
        self.assertIn("thresholds", events[0])
        self.assertIn("claim_id", events[0])

    def test_gateway_event_persisted_to_file(self):
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w") as f:
            path = f.name
        try:
            gw = SPLGateway(audit_log_path=path)
            r = gw.submit(_proj(P_E1))
            gw.emit_claim_nodes(r.candidates)
            with open(path, encoding="utf-8") as f:
                lines = [l for l in f.readlines() if l.strip()]
            self.assertGreater(len(lines), 0)
            event = json.loads(lines[0])
            self.assertIn("decision", event)
            self.assertIn("candidate_id", event)
        finally:
            os.unlink(path)

    def test_audit_log_path_none_disables_file(self):
        """audit_log_path=None must not create any file."""
        gw = SPLGateway(audit_log_path=None)
        r = gw.submit(_proj(P_E1))
        gw.emit_claim_nodes(r.candidates)
        # No file should have been created (test just checks no exception)


class TestSummary(unittest.TestCase):

    def test_summary_counts(self):
        gw = SPLGateway(audit_log_path=None)
        gw.submit_batch([_proj(P_E1), _proj(P_E2), _proj(P_E3),
                         _proj(P_E1, p_illegal=P_E0_ILLEGAL)])
        s = gw.summary()
        self.assertEqual(s["by_status"].get("ready_for_claim", 0), 2)
        self.assertEqual(s["by_status"].get("ambiguous", 0), 1)
        self.assertEqual(s["by_status"].get("structural_violation", 0), 1)

    def test_summary_has_thresholds(self):
        gw = SPLGateway(audit_log_path=None)
        s = gw.summary()
        self.assertIn("tau_1", s["thresholds"])
        self.assertEqual(s["thresholds"]["tau_1"], 0.60)


class TestMakeGateway(unittest.TestCase):

    def test_make_gateway_defaults(self):
        gw = make_gateway(audit_log_path=None)
        self.assertEqual(gw.thresholds.tau_1, 0.60)

    def test_make_gateway_custom(self):
        gw = make_gateway(tau_1=0.75, tau_2=0.15, tau_3=0.70, audit_log_path=None)
        self.assertEqual(gw.thresholds.tau_1, 0.75)

    def test_make_gateway_invalid_raises(self):
        with self.assertRaises(SPLGatewayError):
            make_gateway(tau_2=0.80, tau_3=0.60, audit_log_path=None)


if __name__ == "__main__":
    unittest.main()
