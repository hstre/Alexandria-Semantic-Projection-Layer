"""Clean-room tests for M1--M3: inputs are raw text, never hand-built P_r."""

import os
import sys
import unittest
import uuid

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from spl import EmissionEngine, EmissionRule, SemanticProjection, TripleProbability
from spl_frontend import (
    DistributionalTypeSystem,
    EntityType,
    SemanticCompiler,
)
from spl_gateway import SPLGateway, hash_claim


class TestM1Fragmentation(unittest.TestCase):
    def setUp(self):
        self.compiler = SemanticCompiler()

    def test_one_sentence_can_become_three_units(self):
        text = (
            "The results suggest that remote work may increase productivity, "
            "although the effect varies across sectors."
        )
        result = self.compiler.compile(text, source_ref="doc:wp2")
        self.assertEqual(len(result.units), 3)
        self.assertIn("suggest", result.units[0].source_text)
        self.assertIn("increase", result.units[1].source_text)
        self.assertIn("varies", result.units[2].source_text)
        for unit in result.units:
            self.assertEqual(text[unit.offset_start:unit.offset_end], unit.source_text)

    def test_independent_conjunction_is_split(self):
        result = self.compiler.compile("Heat increases pressure and cold decreases pressure.")
        self.assertEqual(len(result.units), 2)


class TestM2DistributionalTypes(unittest.TestCase):
    def test_classifier_returns_full_normalised_distribution(self):
        distribution = DistributionalTypeSystem().classify(
            "results", "The results suggest that X increases Y", "subject"
        )
        self.assertEqual(set(distribution.probabilities), {kind.value for kind in EntityType})
        self.assertAlmostEqual(sum(distribution.probabilities.values()), 1.0)
        self.assertEqual(distribution.dominant, EntityType.EVIDENCE)

    def test_context_and_role_are_retained_as_uncertainty_not_hidden(self):
        result = SemanticCompiler().compile("The model predicts the election.")
        projection = result.projections[0]
        self.assertIn("subject_type_h_norm", projection.uncertainty)
        self.assertIn("object_type_h_norm", projection.uncertainty)
        self.assertEqual(projection.backend_trace["dominant_types"], ["MODEL", "EVENT"])


class TestM3Projection(unittest.TestCase):
    def setUp(self):
        self.compiler = SemanticCompiler()

    def test_german_and_english_project_to_same_relation_and_entity_ids(self):
        en = self.compiler.compile("Paris is the capital of France.").projections[0]
        de = self.compiler.compile("Paris ist die Hauptstadt von Frankreich.").projections[0]
        self.assertEqual(max(en.P_r, key=en.P_r.get), "capital_of")
        self.assertEqual(en.P_r, de.P_r)
        self.assertEqual(en.triple_distribution[0].subject_id, de.triple_distribution[0].subject_id)
        self.assertEqual(en.triple_distribution[0].object_id, de.triple_distribution[0].object_id)

    def test_dynamic_and_statistical_relations_remain_distinct(self):
        causal = self.compiler.compile("Temperature causes pressure.").projections[0]
        statistical = self.compiler.compile("Temperature correlates with pressure.").projections[0]
        self.assertEqual(max(causal.P_r, key=causal.P_r.get), "causes")
        self.assertEqual(max(statistical.P_r, key=statistical.P_r.get), "correlates_with")
        self.assertGreater(causal.P_family["DYNAMIC"], causal.P_family["STATISTICAL"])
        self.assertGreater(statistical.P_family["STATISTICAL"], statistical.P_family["DYNAMIC"])

    def test_matrix_exposes_illegal_mass(self):
        projection = self.compiler.compile("The rule causes Paris.").projections[0]
        self.assertGreater(projection.p_illegal, 0.50)
        self.assertEqual(projection.P_r, {})
        self.assertEqual(EmissionEngine().emit(projection), [])
        self.assertEqual(projection.emission_rule, EmissionRule.E0)

    def test_structural_violation_reaches_gateway_e0(self):
        projection = self.compiler.compile("The rule causes Paris.").projections[0]
        result = SPLGateway(audit_log_path=None).submit(projection)
        self.assertEqual(result.emission_rule, EmissionRule.E0)

    def test_ambiguous_coordination_is_not_silently_collapsed(self):
        projection = self.compiler.compile(
            "The compound may inhibit or reduce activity under certain conditions."
        ).projections[0]
        self.assertEqual(projection.P_r, {"inhibits": 0.5, "decreases": 0.5})
        self.assertEqual(projection.P_scope, {"under certain conditions": 1.0})
        self.assertEqual(projection.P_modality["possible"], 0.95)
        self.assertEqual(EmissionEngine().emit(projection), [])
        self.assertEqual(projection.emission_rule, EmissionRule.E3)

    def test_projection_contains_normalised_sparse_tensor_and_marginals(self):
        projection = self.compiler.compile("Paris is the capital of France.").projections[0]
        self.assertAlmostEqual(sum(cell.probability for cell in projection.triple_distribution), 1.0)
        self.assertAlmostEqual(sum(projection.P_r.values()), 1.0)
        self.assertEqual(projection.P_subject, {"Paris": 1.0})
        self.assertEqual(projection.P_object, {"France": 1.0})
        self.assertEqual(len(projection.matrix_seal_hash), 64)
        self.assertEqual(projection.backend_trace["backend"], "offline-rule-reference-v1")


class TestTensorEmission(unittest.TestCase):
    def test_emission_uses_tensor_argmax_not_first_surface_candidate(self):
        projection = SemanticProjection(
            projection_id=str(uuid.uuid4()),
            unit_id=str(uuid.uuid4()),
            builder_origin="alpha",
            matrix_version="test",
            P_r={"causes": 0.97, "affects": 0.03},
            subject_candidates=["wrong", "right"],
            object_candidates=["wrong", "target"],
            triple_distribution=[
                TripleProbability("wrong", "affects", "wrong", 0.03),
                TripleProbability("right", "causes", "target", 0.97,
                                  "entity:right", "entity:target", "ENTITY", "ENTITY"),
            ],
            source_ref="doc:tensor",
        )
        candidate = EmissionEngine().emit(projection)[0]
        self.assertEqual((candidate.subject, candidate.object), ("right", "target"))
        self.assertEqual(candidate.subject_id, "entity:right")
        self.assertEqual(candidate.source_ref, "doc:tensor")

    def test_canonical_ids_make_claim_hash_language_independent(self):
        english = hash_claim(
            "Paris", "capital_of", "France",
            subject_id="geo:paris", object_id="geo:france",
        )
        german = hash_claim(
            "Paris", "capital_of", "Frankreich",
            subject_id="geo:paris", object_id="geo:france",
        )
        self.assertEqual(english, german)
        self.assertNotEqual(
            english,
            hash_claim(
                "Paris", "located_in", "France",
                subject_id="geo:paris", object_id="geo:france",
            ),
        )


if __name__ == "__main__":
    unittest.main()
