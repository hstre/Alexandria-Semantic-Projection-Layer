"""
tests/test_spl_rules.py
=======================
Unit tests for SPL emission rules E0–E4 and the full pipeline.

Tests:
    - E0: Structural rejection (p_illegal > τ₀)
    - E1: Singular emission (max_prob > τ₁, H_norm < τ₂)
    - E2: Multiple emission (H_norm ∈ [τ₂, τ₃))
    - E3: Ambiguity block (H_norm ≥ τ₃)
    - E4: Builder divergence (JSD > τ₄) — see also test_jsd.py
    - SPLThresholds validation
    - Full pipeline: "Paris is the capital of France."
    - ClaimCandidateConverter: E3/E4 cannot be converted

Run:
    python -m unittest tests/test_spl_rules.py
"""

import sys
import os
import unittest
import uuid

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import tests._mock_schema as _mock
_mock.setup()

from spl import (
    SemanticUnit, SemanticProjection, ClaimCandidate,
    EmissionEngine, EmissionStatus, EmissionRule,
    SPLThresholds, ClaimCandidateConverter, compute_h_norm,
)

Θ = SPLThresholds()


def _proj(P_r, subjects=("S",), objects=("O",), builder="alpha",
          p_illegal=0.0, unit_id=None):
    return SemanticProjection(
        projection_id=str(uuid.uuid4()),
        unit_id=unit_id or str(uuid.uuid4()),
        builder_origin=builder,
        matrix_version="v2.2.0-TEST",
        P_r=P_r,
        subject_candidates=list(subjects),
        object_candidates=list(objects),
        P_modality={"asserted": 1.0},
        P_category={"dynamic": 1.0},
        p_illegal=p_illegal,
    )


class TestSPLThresholds(unittest.TestCase):

    def test_default_values(self):
        self.assertEqual(Θ.tau_0, 0.50)
        self.assertEqual(Θ.tau_1, 0.60)
        self.assertEqual(Θ.tau_2, 0.25)
        self.assertEqual(Θ.tau_3, 0.65)
        self.assertEqual(Θ.tau_4, 0.40)

    def test_default_is_valid(self):
        self.assertEqual(Θ.validate(), [])

    def test_tau2_must_be_less_than_tau3(self):
        bad = SPLThresholds(tau_2=0.80, tau_3=0.65)
        errors = bad.validate()
        self.assertTrue(len(errors) > 0)

    def test_tau_out_of_range(self):
        self.assertTrue(len(SPLThresholds(tau_0=0.0).validate()) > 0)
        self.assertTrue(len(SPLThresholds(tau_1=1.0).validate()) > 0)


class TestEmissionE0(unittest.TestCase):
    """E0: Structural rejection when p_illegal > τ₀."""

    def setUp(self):
        self.engine = EmissionEngine(Θ)

    def test_e0_triggered(self):
        proj = _proj({"causes": 0.97, "correlates": 0.03}, p_illegal=0.55)
        candidates = self.engine.emit(proj)
        self.assertEqual(proj.status, EmissionStatus.STRUCTURAL_VIOLATION)
        self.assertEqual(proj.emission_rule, EmissionRule.E0)
        self.assertEqual(candidates, [])

    def test_e0_not_triggered_below_threshold(self):
        proj = _proj({"causes": 0.97, "correlates": 0.03}, p_illegal=0.49)
        candidates = self.engine.emit(proj)
        self.assertNotEqual(proj.status, EmissionStatus.STRUCTURAL_VIOLATION)
        self.assertGreater(len(candidates), 0)

    def test_e0_boundary_at_threshold(self):
        """p_illegal == τ₀ exactly does NOT trigger E0 (strictly greater)."""
        proj = _proj({"causes": 0.97, "correlates": 0.03}, p_illegal=0.50)
        self.engine.emit(proj)
        self.assertNotEqual(proj.status, EmissionStatus.STRUCTURAL_VIOLATION)


class TestEmissionE1(unittest.TestCase):
    """E1: Singular emission — max_prob > τ₁ AND H_norm < τ₂."""

    def setUp(self):
        self.engine = EmissionEngine(Θ)

    def test_e1_triggered(self):
        proj = _proj({"causes": 0.97, "correlates": 0.03},
                     subjects=("Paris",), objects=("France",))
        candidates = self.engine.emit(proj)
        self.assertEqual(proj.status, EmissionStatus.READY_FOR_CLAIM)
        self.assertEqual(proj.emission_rule, EmissionRule.E1)
        self.assertEqual(len(candidates), 1)

    def test_e1_candidate_structure(self):
        proj = _proj({"causes": 0.97, "correlates": 0.03},
                     subjects=("Paris",), objects=("France",))
        candidates = self.engine.emit(proj)
        c = candidates[0]
        self.assertEqual(c.relation, "causes")
        self.assertEqual(c.subject, "Paris")
        self.assertEqual(c.object, "France")
        self.assertEqual(c.rank, 1)
        self.assertEqual(c.emission_rule, EmissionRule.E1)
        self.assertGreater(c.relation_score, Θ.tau_1)
        self.assertLess(c.h_norm, Θ.tau_2)

    def test_e1_requires_low_entropy(self):
        """High entropy must prevent E1 even if max_prob > τ₁."""
        # H_norm ≈ 0.558 > τ₂ → E2, not E1
        proj = _proj({"causes": 0.80, "correlates": 0.15, "inhibits": 0.05})
        self.engine.emit(proj)
        self.assertNotEqual(proj.emission_rule, EmissionRule.E1)


class TestEmissionE2(unittest.TestCase):
    """E2: Multiple emission — H_norm ∈ [τ₂, τ₃)."""

    def setUp(self):
        self.engine = EmissionEngine(Θ)

    def test_e2_triggered(self):
        proj = _proj({"suggests": 0.80, "indicates": 0.15, "supports": 0.05})
        candidates = self.engine.emit(proj, k=3)
        self.assertEqual(proj.status, EmissionStatus.READY_FOR_CLAIM)
        self.assertEqual(proj.emission_rule, EmissionRule.E2)
        self.assertEqual(len(candidates), 3)

    def test_e2_rank_ordering(self):
        proj = _proj({"suggests": 0.80, "indicates": 0.15, "supports": 0.05})
        candidates = self.engine.emit(proj, k=3)
        self.assertEqual([c.rank for c in candidates], [1, 2, 3])
        scores = [c.relation_score for c in candidates]
        self.assertTrue(scores[0] >= scores[1] >= scores[2])

    def test_e2_k_limits_candidates(self):
        proj = _proj({"a": 0.80, "b": 0.15, "c": 0.05})
        candidates = self.engine.emit(proj, k=2)
        self.assertEqual(len(candidates), 2)

    def test_e2_entropy_in_range(self):
        proj = _proj({"suggests": 0.80, "indicates": 0.15, "supports": 0.05})
        self.engine.emit(proj)
        self.assertGreaterEqual(proj.h_norm, Θ.tau_2)
        self.assertLess(proj.h_norm, Θ.tau_3)


class TestEmissionE3(unittest.TestCase):
    """E3: Ambiguity block — H_norm ≥ τ₃."""

    def setUp(self):
        self.engine = EmissionEngine(Θ)

    def test_e3_triggered(self):
        proj = _proj({"a": 0.28, "b": 0.27, "c": 0.25, "d": 0.20})
        candidates = self.engine.emit(proj)
        self.assertEqual(proj.status, EmissionStatus.AMBIGUOUS)
        self.assertEqual(proj.emission_rule, EmissionRule.E3)
        self.assertEqual(candidates, [])

    def test_e3_entropy_above_tau3(self):
        proj = _proj({"a": 0.28, "b": 0.27, "c": 0.25, "d": 0.20})
        self.engine.emit(proj)
        self.assertGreaterEqual(proj.h_norm, Θ.tau_3)

    def test_e3_empty_P_r_treated_as_ambiguous(self):
        proj = _proj({})
        candidates = self.engine.emit(proj)
        self.assertEqual(proj.status, EmissionStatus.AMBIGUOUS)
        self.assertEqual(candidates, [])


class TestConverterBoundary(unittest.TestCase):
    """ClaimCandidateConverter: only E1/E2 may be converted."""

    def setUp(self):
        self.converter = ClaimCandidateConverter()
        self.engine = EmissionEngine(Θ)

    def _blocked_candidate(self, rule):
        proj = _proj({"a": 0.28, "b": 0.27, "c": 0.25, "d": 0.20})
        return ClaimCandidate(
            candidate_id=str(uuid.uuid4()),
            projection_id=proj.projection_id,
            unit_id=proj.unit_id,
            source_ref="test",
            subject="X",
            relation="causes",
            object="Y",
            relation_score=0.28,
            emission_rule=rule,
        )

    def test_e3_conversion_raises(self):
        c = self._blocked_candidate(EmissionRule.E3)
        with self.assertRaises(ValueError):
            self.converter.convert(c)

    def test_e4_conversion_raises(self):
        c = self._blocked_candidate(EmissionRule.E4)
        with self.assertRaises(ValueError):
            self.converter.convert(c)

    def test_e1_conversion_succeeds(self):
        proj = _proj({"causes": 0.97, "correlates": 0.03},
                     subjects=("X",), objects=("Y",))
        candidates = self.engine.emit(proj)
        node = self.converter.convert(candidates[0])
        self.assertIsNotNone(node)
        self.assertEqual(node.subject, "X")
        self.assertEqual(node.predicate, "causes")

    def test_convert_batch_skips_blocked(self):
        c_ok = ClaimCandidate(
            candidate_id=str(uuid.uuid4()),
            projection_id=str(uuid.uuid4()),
            unit_id=str(uuid.uuid4()),
            source_ref="",
            subject="A", relation="causes", object="B",
            relation_score=0.97,
            emission_rule=EmissionRule.E1,
        )
        c_bad = self._blocked_candidate(EmissionRule.E3)
        results = self.converter.convert_batch([c_ok, c_bad])
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].subject, "A")


class TestEndToEndPipeline(unittest.TestCase):
    """
    Task 7: Minimal end-to-end test.
    Input: "Paris is the capital of France."
    Expected: ClaimNode(subject="Paris", predicate="capital_of", object="France")
    """

    def test_paris_capital_of_france(self):
        engine    = EmissionEngine(Θ)
        converter = ClaimCandidateConverter()

        # Step 1: SemanticUnit
        unit = SemanticUnit.new(
            source_text="Paris is the capital of France.",
            source_ref="example",
            fragmentation_signal="relational:is",
        )
        self.assertTrue(unit.unit_id)
        self.assertEqual(unit.source_text, "Paris is the capital of France.")

        # Step 2: SemanticProjection
        # NLP backend would produce this distribution;
        # "capital_of" clearly dominant → E1 scenario
        proj = SemanticProjection(
            projection_id=str(uuid.uuid4()),
            unit_id=unit.unit_id,
            builder_origin="alpha",
            matrix_version="v2.2.0-TEST",
            P_r={"capital_of": 0.97, "located_in": 0.03},
            subject_candidates=["Paris"],
            object_candidates=["France"],
            P_modality={"asserted": 1.0},
            P_category={"ontic": 1.0},
        )

        # Step 3: Emission
        candidates = engine.emit(proj)
        self.assertEqual(proj.emission_rule, EmissionRule.E1)
        self.assertEqual(len(candidates), 1)

        c = candidates[0]
        self.assertEqual(c.subject, "Paris")
        self.assertEqual(c.relation, "capital_of")
        self.assertEqual(c.object, "France")

        # Step 4: ClaimNode
        node = converter.convert(c)
        self.assertEqual(node.subject, "Paris")
        self.assertEqual(node.predicate, "capital_of")
        self.assertEqual(node.object, "France")
        self.assertTrue(any("spl:" in ref for ref in node.source_refs))

    def test_multi_unit_sentence(self):
        """
        Sentence: "The results suggest that remote work may increase productivity,
                   although the effect varies across sectors."
        → 3 SemanticUnits → mix of E1 and E2 emissions
        """
        engine = EmissionEngine(Θ)

        units_and_P_r = [
            ("results suggest [remote work increases productivity]",
             {"suggests": 0.80, "indicates": 0.15, "supports": 0.05},
             ["results"], ["remote work increases productivity"]),
            ("remote work may increase productivity",
             {"may_increase": 0.97, "correlates_with": 0.03},
             ["remote work"], ["productivity"]),
            ("effect varies across sectors",
             {"varies_across": 0.75, "depends_on": 0.20, "correlates_with": 0.05},
             ["effect"], ["sectors"]),
        ]

        all_candidates = []
        rules = []
        for text, P_r, subjs, objs in units_and_P_r:
            unit = SemanticUnit.new(source_text=text, source_ref="wp2-example")
            proj = SemanticProjection(
                projection_id=str(uuid.uuid4()),
                unit_id=unit.unit_id,
                builder_origin="alpha",
                matrix_version="v2.2.0-TEST",
                P_r=P_r,
                subject_candidates=subjs,
                object_candidates=objs,
                P_modality={"suggested": 1.0},
                P_category={"dynamic": 1.0},
            )
            candidates = engine.emit(proj, k=3)
            all_candidates.extend(candidates)
            rules.append(proj.emission_rule)

        self.assertEqual(rules[0], EmissionRule.E2)  # "suggest" — multi
        self.assertEqual(rules[1], EmissionRule.E1)  # "may_increase" — singular
        self.assertEqual(rules[2], EmissionRule.E2)  # "varies_across" — multi
        self.assertGreaterEqual(len(all_candidates), 3)


if __name__ == "__main__":
    unittest.main()
