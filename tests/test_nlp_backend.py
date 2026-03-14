"""
Tests for nlp_backend.py
========================

Uses a deterministic mock for SentenceTransformer so tests are fast
and reproducible without downloading any model.
"""

import unittest
from unittest.mock import MagicMock, patch

import numpy as np

from nlp_backend import (
    CATEGORY_ANCHORS,
    MATRIX_VERSION,
    MODALITY_ANCHORS,
    RELATION_ANCHORS,
    SPLNLPBackend,
    _cosine_sim,
    _heuristic_candidates,
    _softmax,
    make_dual_backends,
)
from spl import EmissionEngine, SemanticUnit


# ---------------------------------------------------------------------------
#  Test helpers
# ---------------------------------------------------------------------------

def _phrase_seed(phrase: str) -> int:
    """Deterministic seed derived from a phrase string."""
    return sum(ord(c) for c in phrase) % (2 ** 31)


def _make_encode_fn(dim: int = 32):
    """
    Returns an encode() side-effect that produces deterministic unit-norm
    embeddings.  Each phrase gets its own seed so different anchor phrases
    produce different embeddings.
    """
    def _encode(phrases, convert_to_numpy=True):
        embs = np.zeros((len(phrases), dim))
        for i, phrase in enumerate(phrases):
            rng = np.random.default_rng(_phrase_seed(phrase))
            v   = rng.standard_normal(dim)
            embs[i] = v / np.linalg.norm(v)
        return embs
    return _encode


def _make_backend(
    temperature: float = 0.5,
    builder:     str   = "alpha",
    dim:         int   = 32,
) -> SPLNLPBackend:
    """Create an SPLNLPBackend with a mocked SentenceTransformer."""
    mock_model = MagicMock()
    mock_model.encode.side_effect = _make_encode_fn(dim)
    with patch("nlp_backend.SentenceTransformer", return_value=mock_model):
        backend = SPLNLPBackend(temperature=temperature, builder_origin=builder)
    return backend


# ---------------------------------------------------------------------------
#  Unit tests: _softmax
# ---------------------------------------------------------------------------

class TestSoftmax(unittest.TestCase):

    def test_sums_to_one(self):
        result = _softmax({"a": 0.9, "b": 0.5, "c": 0.1}, temperature=1.0)
        self.assertAlmostEqual(sum(result.values()), 1.0, places=10)

    def test_all_keys_present(self):
        scores = {"x": 0.3, "y": -0.1, "z": 0.7}
        result = _softmax(scores, temperature=0.5)
        self.assertEqual(set(result.keys()), set(scores.keys()))

    def test_argmax_preserved(self):
        scores = {"a": 0.9, "b": 0.2, "c": -0.5}
        result = _softmax(scores, temperature=0.5)
        self.assertEqual(max(result, key=result.get), "a")

    def test_all_non_negative(self):
        scores = {"a": -0.9, "b": -0.5, "c": -0.1}
        result = _softmax(scores, temperature=1.0)
        self.assertTrue(all(v >= 0 for v in result.values()))

    def test_low_temperature_sharpens(self):
        scores = {"a": 0.8, "b": 0.2}
        hot  = _softmax(scores, temperature=5.0)
        cold = _softmax(scores, temperature=0.1)
        self.assertGreater(cold["a"], hot["a"])

    def test_single_key(self):
        result = _softmax({"only": 0.5}, temperature=1.0)
        self.assertAlmostEqual(result["only"], 1.0, places=10)

    def test_negative_scores_valid(self):
        scores = {"a": -0.5, "b": -0.3, "c": -0.1}
        result = _softmax(scores, temperature=1.0)
        self.assertAlmostEqual(sum(result.values()), 1.0, places=10)


# ---------------------------------------------------------------------------
#  Unit tests: _cosine_sim
# ---------------------------------------------------------------------------

class TestCosineSim(unittest.TestCase):

    def test_identical_vectors(self):
        v = np.array([1.0, 2.0, 3.0])
        self.assertAlmostEqual(_cosine_sim(v, v), 1.0, places=6)

    def test_orthogonal_vectors(self):
        a = np.array([1.0, 0.0])
        b = np.array([0.0, 1.0])
        self.assertAlmostEqual(_cosine_sim(a, b), 0.0, places=6)

    def test_antiparallel_vectors(self):
        v = np.array([1.0, 0.0])
        self.assertAlmostEqual(_cosine_sim(v, -v), -1.0, places=6)

    def test_zero_vector_returns_zero(self):
        a = np.zeros(3)
        b = np.array([1.0, 2.0, 3.0])
        self.assertEqual(_cosine_sim(a, b), 0.0)

    def test_range(self):
        rng = np.random.default_rng(0)
        for _ in range(20):
            a = rng.standard_normal(16)
            b = rng.standard_normal(16)
            s = _cosine_sim(a, b)
            self.assertGreaterEqual(s, -1.0 - 1e-9)
            self.assertLessEqual(s,    1.0 + 1e-9)


# ---------------------------------------------------------------------------
#  Unit tests: _heuristic_candidates
# ---------------------------------------------------------------------------

class TestHeuristicCandidates(unittest.TestCase):

    def test_capital_of(self):
        subj, obj = _heuristic_candidates("Paris is the capital of France.")
        self.assertIn("Paris", subj[0])
        self.assertIn("France", obj[0])

    def test_causes(self):
        subj, obj = _heuristic_candidates("Smoking causes lung cancer.")
        self.assertIn("Smoking", subj[0])
        self.assertIn("lung cancer", obj[0])

    def test_increases(self):
        subj, obj = _heuristic_candidates("Exercise increases cardiovascular fitness.")
        self.assertTrue(len(subj) > 0 and len(obj) > 0)

    def test_suggests(self):
        subj, obj = _heuristic_candidates("The data suggests a correlation.")
        self.assertIn("The data", subj[0])
        self.assertIn("a correlation", obj[0])

    def test_no_split_fallback_non_empty(self):
        subj, obj = _heuristic_candidates("xyzzy quux blorb")
        self.assertTrue(len(subj) > 0)
        self.assertTrue(len(obj) > 0)

    def test_contradicts(self):
        subj, obj = _heuristic_candidates("Study A contradicts Study B.")
        self.assertIn("Study A", subj[0])
        self.assertIn("Study B", obj[0])

    def test_defines(self):
        subj, obj = _heuristic_candidates("This paper defines entropy as disorder.")
        self.assertTrue(len(subj) > 0 and len(obj) > 0)


# ---------------------------------------------------------------------------
#  Unit tests: SPLNLPBackend construction
# ---------------------------------------------------------------------------

class TestBackendInit(unittest.TestCase):

    def test_invalid_builder_origin_raises(self):
        mock_model = MagicMock()
        mock_model.encode.side_effect = _make_encode_fn()
        with patch("nlp_backend.SentenceTransformer", return_value=mock_model):
            with self.assertRaises(ValueError):
                SPLNLPBackend(builder_origin="gamma")

    def test_invalid_temperature_raises(self):
        mock_model = MagicMock()
        mock_model.encode.side_effect = _make_encode_fn()
        with patch("nlp_backend.SentenceTransformer", return_value=mock_model):
            with self.assertRaises(ValueError):
                SPLNLPBackend(temperature=0.0)
            with self.assertRaises(ValueError):
                SPLNLPBackend(temperature=-1.0)

    def test_alpha_builder(self):
        b = _make_backend(builder="alpha")
        self.assertEqual(b.builder_origin, "alpha")

    def test_beta_builder(self):
        b = _make_backend(builder="beta")
        self.assertEqual(b.builder_origin, "beta")

    def test_matrix_version(self):
        b = _make_backend()
        self.assertEqual(b.matrix_version, MATRIX_VERSION)


# ---------------------------------------------------------------------------
#  Unit tests: create_unit
# ---------------------------------------------------------------------------

class TestCreateUnit(unittest.TestCase):

    def setUp(self):
        self.backend = _make_backend()

    def test_returns_semantic_unit(self):
        unit = self.backend.create_unit("Paris is the capital of France.")
        self.assertIsInstance(unit, SemanticUnit)

    def test_source_text_preserved(self):
        text = "Remote work increases productivity."
        unit = self.backend.create_unit(text)
        self.assertEqual(unit.source_text, text)

    def test_source_ref_preserved(self):
        unit = self.backend.create_unit("x", source_ref="WP2")
        self.assertEqual(unit.source_ref, "WP2")

    def test_offset_end_defaults_to_len(self):
        text = "hello world"
        unit = self.backend.create_unit(text)
        self.assertEqual(unit.offset_end, len(text))

    def test_fragmentation_signal_preserved(self):
        unit = self.backend.create_unit("x", fragmentation_signal="modal:may")
        self.assertEqual(unit.fragmentation_signal, "modal:may")


# ---------------------------------------------------------------------------
#  Unit tests: project()
# ---------------------------------------------------------------------------

class TestProject(unittest.TestCase):

    def setUp(self):
        self.backend = _make_backend()

    def _proj(self, text: str = "Paris is the capital of France."):
        return self.backend.project_text(text)

    # -- P_r ---------------------------------------------------------------

    def test_P_r_sums_to_one(self):
        self.assertAlmostEqual(sum(self._proj().P_r.values()), 1.0, places=10)

    def test_P_r_has_all_relations(self):
        self.assertEqual(set(self._proj().P_r.keys()), set(RELATION_ANCHORS.keys()))

    def test_P_r_all_non_negative(self):
        self.assertTrue(all(v >= 0 for v in self._proj().P_r.values()))

    def test_P_r_has_clear_argmax(self):
        # Argmax should exist and sum to less than 1 (not degenerate)
        p = self._proj().P_r
        max_p = max(p.values())
        self.assertGreater(max_p, 0.0)
        self.assertLessEqual(max_p, 1.0)

    # -- P_category --------------------------------------------------------

    def test_P_category_sums_to_one(self):
        self.assertAlmostEqual(sum(self._proj().P_category.values()), 1.0, places=10)

    def test_P_category_has_all_keys(self):
        self.assertEqual(set(self._proj().P_category.keys()), set(CATEGORY_ANCHORS.keys()))

    def test_P_category_all_non_negative(self):
        self.assertTrue(all(v >= 0 for v in self._proj().P_category.values()))

    # -- P_modality --------------------------------------------------------

    def test_P_modality_sums_to_one(self):
        self.assertAlmostEqual(sum(self._proj().P_modality.values()), 1.0, places=10)

    def test_P_modality_has_all_keys(self):
        self.assertEqual(set(self._proj().P_modality.keys()), set(MODALITY_ANCHORS.keys()))

    def test_P_modality_all_non_negative(self):
        self.assertTrue(all(v >= 0 for v in self._proj().P_modality.values()))

    # -- metadata ----------------------------------------------------------

    def test_builder_origin_written(self):
        self.assertEqual(self._proj().builder_origin, "alpha")

    def test_matrix_version_written(self):
        self.assertEqual(self._proj().matrix_version, MATRIX_VERSION)

    def test_p_illegal_is_zero(self):
        self.assertEqual(self._proj().p_illegal, 0.0)

    def test_projection_id_is_uuid(self):
        import uuid as _uuid
        proj = self._proj()
        _uuid.UUID(proj.projection_id)     # raises ValueError if not a valid UUID

    def test_unit_id_linked(self):
        unit = self.backend.create_unit("X defines Y.")
        proj = self.backend.project(unit)
        self.assertEqual(proj.unit_id, unit.unit_id)

    def test_subject_candidates_non_empty(self):
        self.assertTrue(len(self._proj().subject_candidates) > 0)

    def test_object_candidates_non_empty(self):
        self.assertTrue(len(self._proj().object_candidates) > 0)

    # -- temperature effect ------------------------------------------------

    def test_lower_temperature_sharpens_P_r(self):
        hot  = _make_backend(temperature=5.0)
        cold = _make_backend(temperature=0.1)
        text = "Smoking causes lung cancer."
        p_hot  = max(hot.project_text(text).P_r.values())
        p_cold = max(cold.project_text(text).P_r.values())
        self.assertGreater(p_cold, p_hot)


# ---------------------------------------------------------------------------
#  Unit tests: make_dual_backends
# ---------------------------------------------------------------------------

class TestDualBackends(unittest.TestCase):

    def _make_pair(self, T_alpha=0.5, T_beta=0.8):
        mock_model = MagicMock()
        mock_model.encode.side_effect = _make_encode_fn()
        with patch("nlp_backend.SentenceTransformer", return_value=mock_model):
            return make_dual_backends(
                temperature_alpha=T_alpha,
                temperature_beta=T_beta,
            )

    def test_returns_two_backends(self):
        alpha, beta = self._make_pair()
        self.assertIsInstance(alpha, SPLNLPBackend)
        self.assertIsInstance(beta,  SPLNLPBackend)

    def test_builder_origins(self):
        alpha, beta = self._make_pair()
        self.assertEqual(alpha.builder_origin, "alpha")
        self.assertEqual(beta.builder_origin,  "beta")

    def test_temperatures_set(self):
        alpha, beta = self._make_pair(T_alpha=0.3, T_beta=1.2)
        self.assertEqual(alpha.temperature, 0.3)
        self.assertEqual(beta.temperature,  1.2)

    def test_same_matrix_version(self):
        alpha, beta = self._make_pair()
        self.assertEqual(alpha.matrix_version, beta.matrix_version)


# ---------------------------------------------------------------------------
#  Integration tests: NLP backend → EmissionEngine
# ---------------------------------------------------------------------------

class TestEmissionIntegration(unittest.TestCase):
    """
    Verify that projections produced by SPLNLPBackend are structurally
    valid inputs for EmissionEngine.  We test structural properties only —
    specific emission rules depend on the (mocked) embeddings.
    """

    def setUp(self):
        self.backend = _make_backend()
        self.engine  = EmissionEngine()

    def test_emission_does_not_raise(self):
        proj = self.backend.project_text("Paris is the capital of France.")
        candidates = self.engine.emit(proj)  # must not raise

    def test_emission_rule_is_not_e0(self):
        # p_illegal=0.0 → E0 must never fire
        proj = self.backend.project_text("Remote work increases productivity.")
        self.engine.emit(proj)
        self.assertNotEqual(proj.emission_rule.value, "E0")

    def test_h_norm_in_range_after_emission(self):
        proj = self.backend.project_text("Smoking causes lung cancer.")
        self.engine.emit(proj)
        self.assertGreaterEqual(proj.h_norm, 0.0)
        self.assertLessEqual(proj.h_norm,    1.0)

    def test_status_set_after_emission(self):
        from spl import EmissionStatus
        proj = self.backend.project_text("Antibiotics prevent bacterial infection.")
        self.engine.emit(proj)
        self.assertIsInstance(proj.status, EmissionStatus)
        self.assertNotEqual(proj.status, EmissionStatus.PROJECTED)

    def test_candidates_have_valid_relation(self):
        proj = self.backend.project_text("Exercise increases cardiovascular fitness.")
        candidates = self.engine.emit(proj)
        for c in candidates:
            self.assertIn(c.relation, RELATION_ANCHORS)

    def test_candidate_scores_in_range(self):
        proj = self.backend.project_text("The data supports the hypothesis.")
        candidates = self.engine.emit(proj)
        for c in candidates:
            self.assertGreaterEqual(c.relation_score, 0.0)
            self.assertLessEqual(c.relation_score,    1.0)

    def test_dual_builder_jsd(self):
        """JSD between alpha and beta projections must be in [0, 1]."""
        from spl import compute_jsd

        alpha = _make_backend(temperature=0.5)
        beta  = _make_backend(temperature=2.0, builder="beta")
        text  = "GDP correlates with life expectancy."
        proj_a = alpha.project_text(text)
        proj_b = beta.project_text(text)
        jsd = compute_jsd(proj_a.P_r, proj_b.P_r)
        self.assertGreaterEqual(jsd, 0.0)
        self.assertLessEqual(jsd,    1.0)


if __name__ == "__main__":
    unittest.main()
