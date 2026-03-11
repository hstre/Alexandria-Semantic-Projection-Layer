"""
tests/test_entropy.py
=====================
Unit tests for H_norm (normalised Shannon entropy) — WP2 §3.3.3, §7.1

Tests:
    - Point mass (H_norm = 0)
    - Uniform distribution (H_norm = 1)
    - Single-element distribution (H_norm = 0 by convention)
    - Empty distribution (H_norm = 0 by convention)
    - Emission region boundaries: E1 < τ₂, E2 ∈ [τ₂, τ₃), E3 ≥ τ₃
    - Monotonicity: spreading mass increases entropy
    - Normalisation invariant: H_norm ∈ [0, 1]

Run:
    python -m unittest tests/test_entropy.py
"""

import sys
import os
import unittest
import math

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import tests._mock_schema as _mock
_mock.setup()

from spl import compute_h_norm, SPLThresholds

Θ = SPLThresholds()


class TestHNormBoundaryValues(unittest.TestCase):
    """Boundary values: point mass and uniform distribution."""

    def test_point_mass_is_zero(self):
        """H_norm of a degenerate (certain) distribution must be 0."""
        self.assertAlmostEqual(compute_h_norm({"causes": 1.0}), 0.0)

    def test_single_element_is_zero(self):
        """Single-element distributions: log(1) = 0, so H_norm = 0."""
        self.assertAlmostEqual(compute_h_norm({"only_relation": 0.999}), 0.0)

    def test_empty_distribution_is_zero(self):
        """Empty P_r: convention H_norm = 0 (no uncertainty)."""
        self.assertEqual(compute_h_norm({}), 0.0)

    def test_uniform_n2_is_one(self):
        """Two-relation uniform distribution: H_norm = 1."""
        self.assertAlmostEqual(compute_h_norm({"a": 0.5, "b": 0.5}), 1.0)

    def test_uniform_n4_is_one(self):
        """Four-relation uniform distribution: H_norm = 1."""
        p = {f"r{i}": 0.25 for i in range(4)}
        self.assertAlmostEqual(compute_h_norm(p), 1.0, places=10)

    def test_uniform_n10_is_one(self):
        """Ten-relation uniform distribution: H_norm = 1."""
        p = {f"r{i}": 0.1 for i in range(10)}
        self.assertAlmostEqual(compute_h_norm(p), 1.0, places=10)


class TestHNormEmissionRegions(unittest.TestCase):
    """
    H_norm partitions the simplex into emission regions (WP2 §3.4.2):
        H_norm < τ₂          → vertex neighbourhood (E1)
        τ₂ ≤ H_norm < τ₃    → face region (E2)
        H_norm ≥ τ₃          → interior / centroid (E3 block)
    """

    def test_e1_region_below_tau2(self):
        """Very dominant distribution must be in E1 region."""
        p = {"causes": 0.97, "correlates": 0.03}
        h = compute_h_norm(p)
        self.assertLess(h, Θ.tau_2, f"H={h:.4f} must be < τ₂={Θ.tau_2}")

    def test_e2_region_between_tau2_and_tau3(self):
        """Moderate distribution must be in E2 region."""
        p = {"suggests": 0.80, "indicates": 0.15, "supports": 0.05}
        h = compute_h_norm(p)
        self.assertGreaterEqual(h, Θ.tau_2,
                                f"H={h:.4f} must be ≥ τ₂={Θ.tau_2}")
        self.assertLess(h, Θ.tau_3, f"H={h:.4f} must be < τ₃={Θ.tau_3}")

    def test_e3_region_at_or_above_tau3(self):
        """Near-uniform distribution must be in E3 region."""
        p = {"a": 0.28, "b": 0.27, "c": 0.25, "d": 0.20}
        h = compute_h_norm(p)
        self.assertGreaterEqual(h, Θ.tau_3,
                                f"H={h:.4f} must be ≥ τ₃={Θ.tau_3}")


class TestHNormMonotonicity(unittest.TestCase):
    """Entropy increases as mass spreads — geometric intuition (WP2 §3.4.2)."""

    def test_spreading_mass_increases_entropy(self):
        """Moving probability from dominant to others must raise H_norm."""
        p_concentrated = {"a": 0.97, "b": 0.03}
        p_spread       = {"a": 0.60, "b": 0.40}
        self.assertLess(compute_h_norm(p_concentrated),
                        compute_h_norm(p_spread))

    def test_three_relation_ordering(self):
        """More concentrated → lower H_norm."""
        h1 = compute_h_norm({"a": 0.97, "b": 0.02, "c": 0.01})
        h2 = compute_h_norm({"a": 0.70, "b": 0.20, "c": 0.10})
        h3 = compute_h_norm({"a": 0.40, "b": 0.35, "c": 0.25})
        self.assertLess(h1, h2)
        self.assertLess(h2, h3)


class TestHNormInvariant(unittest.TestCase):
    """H_norm ∈ [0, 1] for any valid probability distribution."""

    def test_range_random_distributions(self):
        """H_norm must always be in [0, 1]."""
        import random
        random.seed(2024)
        for _ in range(50):
            n = random.randint(2, 12)
            vals = [random.random() for _ in range(n)]
            s = sum(vals)
            p = {f"r{i}": v / s for i, v in enumerate(vals)}
            h = compute_h_norm(p)
            self.assertGreaterEqual(h, 0.0 - 1e-9)
            self.assertLessEqual(h, 1.0 + 1e-9)

    def test_normalisation_by_log_n(self):
        """Manual verification of H_norm = H / log2(n)."""
        p = {"a": 0.6, "b": 0.4}
        h_raw = -(0.6 * math.log2(0.6) + 0.4 * math.log2(0.4))
        expected = h_raw / math.log2(2)
        self.assertAlmostEqual(compute_h_norm(p), expected, places=10)


if __name__ == "__main__":
    unittest.main()
