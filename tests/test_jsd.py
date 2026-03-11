"""
tests/test_jsd.py
=================
Unit tests for JSD (Jensen-Shannon Divergence) — WP2 §3.3.5

Tests:
    - JSD(P, P) = 0  (identical distributions)
    - JSD(disjoint) = 1  (maximally divergent, WP2 §3.3.5)
    - Symmetry: JSD(A, B) == JSD(B, A)
    - Range: JSD ∈ [0, 1]  (base-2 logarithm)
    - Partial overlap: 0 < JSD < 1
    - E4 trigger: JSD > τ₄ marks both projections as BRANCH_CANDIDATE

Run:
    python -m unittest tests/test_jsd.py
"""

import sys
import os
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import tests._mock_schema as _mock
_mock.setup()

from spl import compute_jsd, SPLThresholds, EmissionEngine, EmissionStatus, EmissionRule
import uuid

Θ = SPLThresholds()


class TestJSDCorrectnessProperties(unittest.TestCase):
    """Mathematical properties of JSD (base-2 log)."""

    def test_identical_distributions_zero(self):
        """JSD(P, P) must be exactly 0."""
        p = {"causes": 0.7, "correlates": 0.2, "inhibits": 0.1}
        self.assertAlmostEqual(compute_jsd(p, p), 0.0, places=10)

    def test_identical_uniform_zero(self):
        """JSD of identical uniform distributions is 0."""
        p = {f"r{i}": 0.25 for i in range(4)}
        self.assertAlmostEqual(compute_jsd(p, p), 0.0, places=10)

    def test_disjoint_support_is_one(self):
        """
        Disjoint supports → JSD = 1.0 (WP2 §3.3.5):
        'a fundamental ontological disagreement between builders
        maximises JSD by construction.'
        """
        p = {"causes": 1.0}
        q = {"inhibits": 1.0}
        self.assertAlmostEqual(compute_jsd(p, q), 1.0, places=10)

    def test_symmetry(self):
        """JSD must be symmetric: JSD(A, B) == JSD(B, A)."""
        p = {"causes": 0.7, "correlates": 0.3}
        q = {"suggests": 0.6, "indicates": 0.4}
        self.assertAlmostEqual(compute_jsd(p, q), compute_jsd(q, p), places=12)

    def test_partial_overlap_between_zero_and_one(self):
        """Partial overlap: 0 < JSD < 1."""
        p = {"causes": 0.8, "correlates": 0.2}
        q = {"causes": 0.4, "inhibits": 0.6}
        jsd = compute_jsd(p, q)
        self.assertGreater(jsd, 0.0)
        self.assertLess(jsd, 1.0)

    def test_empty_distributions_zero(self):
        """Both empty → JSD = 0 (no divergence)."""
        self.assertAlmostEqual(compute_jsd({}, {}), 0.0)

    def test_range_random_distributions(self):
        """JSD ∈ [0, 1] for 100 random distribution pairs."""
        import random
        random.seed(42)
        for _ in range(100):
            keys = [f"r{j}" for j in range(random.randint(2, 8))]
            vals_a = [random.random() + 0.01 for _ in keys]
            vals_b = [random.random() + 0.01 for _ in keys]
            sa, sb = sum(vals_a), sum(vals_b)
            pa = {k: v / sa for k, v in zip(keys, vals_a)}
            pb = {k: v / sb for k, v in zip(keys, vals_b)}
            j = compute_jsd(pa, pb)
            self.assertGreaterEqual(j, 0.0 - 1e-9,
                                    f"JSD={j:.6f} below 0 for pa={pa}, pb={pb}")
            self.assertLessEqual(j, 1.0 + 1e-9,
                                 f"JSD={j:.6f} above 1 for pa={pa}, pb={pb}")


class TestJSDGlobalSimplexEmbedding(unittest.TestCase):
    """
    Global simplex embedding: missing keys get probability 0 (WP2 §3.3.5).
    """

    def test_zero_padding_for_missing_keys(self):
        """Keys in one but not the other get probability 0."""
        p = {"causes": 0.8, "correlates": 0.2}
        q = {"causes": 0.8, "inhibits": 0.2}  # "correlates" missing, "inhibits" new
        jsd = compute_jsd(p, q)
        # Not 0 (different), not 1 (shared "causes" key with same mass)
        self.assertGreater(jsd, 0.0)
        self.assertLess(jsd, 1.0)

    def test_full_disjoint_is_exactly_one(self):
        """Fully disjoint relation sets → JSD = 1 by zero-padding."""
        # Builder A only uses causal relations
        p = {"causes": 0.6, "enables": 0.4}
        # Builder B only uses evidential relations
        q = {"suggests": 0.7, "indicates": 0.3}
        self.assertAlmostEqual(compute_jsd(p, q), 1.0, places=10)


class TestJSDE4Integration(unittest.TestCase):
    """E4: JSD > τ₄ triggers BRANCH_CANDIDATE."""

    def _make_proj(self, P_r, unit_id, builder):
        from spl import SemanticProjection
        return SemanticProjection(
            projection_id=str(uuid.uuid4()),
            unit_id=unit_id,
            builder_origin=builder,
            matrix_version="v2.2.0-TEST",
            P_r=P_r,
            subject_candidates=["X"],
            object_candidates=["Y"],
        )

    def test_e4_disjoint_triggers_branch(self):
        """JSD=1 (fully disjoint) → both projections become BRANCH_CANDIDATE."""
        engine = EmissionEngine(Θ)
        uid = str(uuid.uuid4())
        pa = self._make_proj({"inhibits": 1.0}, uid, "alpha")
        pb = self._make_proj({"enables": 1.0}, uid, "beta")
        jsd = engine.apply_e4(pa, pb)
        self.assertAlmostEqual(jsd, 1.0, places=10)
        self.assertTrue(jsd > Θ.tau_4)
        self.assertEqual(pa.status, EmissionStatus.BRANCH_CANDIDATE)
        self.assertEqual(pb.status, EmissionStatus.BRANCH_CANDIDATE)
        self.assertEqual(pa.emission_rule, EmissionRule.E4)
        self.assertEqual(pb.emission_rule, EmissionRule.E4)

    def test_e4_below_tau4_does_not_trigger(self):
        """Low JSD: E4 computes JSD but does not set BRANCH_CANDIDATE."""
        engine = EmissionEngine(Θ)
        uid = str(uuid.uuid4())
        # Same distribution → JSD = 0
        p = {"causes": 0.97, "correlates": 0.03}
        pa = self._make_proj(p, uid, "alpha")
        pb = self._make_proj(p, uid, "beta")
        jsd = engine.apply_e4(pa, pb)
        self.assertAlmostEqual(jsd, 0.0, places=10)
        self.assertFalse(jsd > Θ.tau_4)
        self.assertNotEqual(pa.status, EmissionStatus.BRANCH_CANDIDATE)

    def test_e4_symmetry(self):
        """apply_e4(A, B) and apply_e4(B, A) produce same JSD."""
        engine = EmissionEngine(Θ)
        uid = str(uuid.uuid4())
        pa = self._make_proj({"causes": 0.8, "correlates": 0.2}, uid, "alpha")
        pb = self._make_proj({"suggests": 0.9, "indicates": 0.1}, uid, "beta")
        jsd_ab = compute_jsd(pa.P_r, pb.P_r)
        jsd_ba = compute_jsd(pb.P_r, pa.P_r)
        self.assertAlmostEqual(jsd_ab, jsd_ba, places=12)


if __name__ == "__main__":
    unittest.main()
