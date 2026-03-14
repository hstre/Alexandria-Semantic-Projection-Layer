"""
Alexandria SPL — NLP Backend (sentence-transformers)
=====================================================

Computes SemanticProjection from raw text using anchor embeddings.

Architecture
------------
Each relation r ∈ ℛ is paired with an anchor phrase that captures its
semantics in natural language.  The backend:

    1. Encodes all anchor phrases once at init-time → cached embeddings.
    2. For each incoming text *t*:
       a. Embeds *t*                   →  e_t
       b. Cosine similarity:            sim(r) = cos(e_t, anchor_emb[r])
       c. Temperature-scaled softmax    →  P_r  (normalised, sums to 1.0)
    3. Repeats for P_category and P_modality using their own anchor sets.
    4. Returns a SemanticProjection ready for EmissionEngine / SPLGateway.

Usage
-----
    from nlp_backend import SPLNLPBackend

    backend = SPLNLPBackend()
    proj = backend.project_text("Paris is the capital of France.")
    # proj.P_r → {"capital_of": 0.68, "is_a": 0.11, ...}

Dual-builder
------------
    alpha, beta = make_dual_backends()          # factory helper
    # or:
    alpha = SPLNLPBackend(builder_origin="alpha", temperature=0.5)
    beta  = SPLNLPBackend(builder_origin="beta",  temperature=0.8)

Temperature guide
-----------------
    T → 0    sharper P_r  (higher max_prob, lower H_norm  → favours E1)
    T → ∞    flatter P_r  (lower max_prob,  higher H_norm → favours E3)
    T = 0.5  default: good balance for all-MiniLM-L6-v2 cosine range
"""

from __future__ import annotations

import re
import uuid
from typing import Optional

import numpy as np
from sentence_transformers import SentenceTransformer

from spl import SemanticProjection, SemanticUnit


# ---------------------------------------------------------------------------
#  Relation matrix ℛ  —  v2.2.0-SML
# ---------------------------------------------------------------------------

MATRIX_VERSION = "v2.2.0-SML"

#: Anchor phrases for each relation in ℛ.
#: Each phrase is written to sit close to the relation's typical
#: natural-language realisation in sentence-transformer embedding space.
RELATION_ANCHORS: dict[str, str] = {
    # ── causal ────────────────────────────────────────────────────────────
    "causes":               "directly causes or produces as a result",
    "enables":              "makes possible or enables to happen",
    "inhibits":             "inhibits or suppresses the occurrence of",
    "prevents":             "prevents or blocks from happening",
    "increases":            "increases or raises the level of",
    "decreases":            "decreases or reduces the amount of",
    # ── correlational ─────────────────────────────────────────────────────
    "correlates_with":      "statistically correlates with or co-varies",
    "inversely_correlates": "inversely correlates or is negatively associated with",
    "cooccurs_with":        "frequently co-occurs or appears together with",
    # ── ontological ───────────────────────────────────────────────────────
    "is_a":                 "is a type of or is classified as",
    "is_part_of":           "is a component or part of",
    "defines":              "defines or is defined as meaning",
    "is_example_of":        "is a concrete example or instance of",
    "is_property_of":       "is a property or attribute of",
    "capital_of":           "is the capital city or administrative seat of",
    # ── epistemic ─────────────────────────────────────────────────────────
    "supports":             "provides evidence that supports or confirms",
    "contradicts":          "contradicts or provides evidence against",
    "suggests":             "suggests or implies that this may be true",
    "questions":            "raises questions about or challenges the validity of",
    "indicates":            "indicates or serves as a signal for",
    "measures":             "measures or quantifies the degree of",
    # ── temporal ──────────────────────────────────────────────────────────
    "precedes":             "temporally precedes or comes before",
    "follows":              "follows as a consequence or comes after",
    "cooccurs_at":          "occurs simultaneously or at the same time as",
    # ── modal ─────────────────────────────────────────────────────────────
    "may_affect":           "may affect or could potentially influence",
    "likely_causes":        "likely causes or probably leads to",
    "possibly_relates":     "possibly relates to or might be connected with",
}

CATEGORY_ANCHORS: dict[str, str] = {
    "dynamic":     "dynamic causal process involving change or action over time",
    "statistical": "statistical pattern or empirical correlation observed in data",
    "epistemic":   "epistemic claim about knowledge evidence or belief states",
    "model":       "theoretical model prediction or computational simulation result",
    "normative":   "normative statement about what ought to be or is required",
    "ontic":       "ontological fact about existence classification or identity",
}

MODALITY_ANCHORS: dict[str, str] = {
    "asserted":     "is clearly and definitively stated as an established fact",
    "suggested":    "is suggested or presented as likely or probable",
    "hypothesized": "is hypothesized or proposed as a speculative theory",
    "possible":     "is stated as a mere possibility that might be the case",
}

# Common relational verbs/phrases used by the regex fallback to split
# subject (left) from object (right)
_SPLIT_RE = re.compile(
    r"\b(?:is\s+the\s+capital\s+of"
    r"|is\s+(?:a|an|part\s+of|example\s+of)"
    r"|are|was|were|has|have|had"
    r"|causes?|enables?|increases?|decreases?|inhibits?|prevents?"
    r"|supports?|contradicts?|suggests?|indicates?|questions?|measures?"
    r"|defines?|precedes?|follows?"
    r"|correlates?\s+with|co.?occurs?\s+with"
    r")\b",
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------
#  Math helpers
# ---------------------------------------------------------------------------

def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two 1-D vectors; returns 0.0 for zero vectors."""
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / denom) if denom > 0.0 else 0.0


def _softmax(scores: dict[str, float], temperature: float) -> dict[str, float]:
    """
    Temperature-scaled softmax over a dict of raw scores.

    Returns a valid probability distribution (all values ≥ 0, sum = 1.0).
    Handles negative scores (cosine similarity range [-1, 1]) correctly.
    """
    keys = list(scores)
    arr  = np.array([scores[k] for k in keys], dtype=np.float64)
    arr  = arr / max(temperature, 1e-9)
    arr -= arr.max()                    # numerical stability
    exp  = np.exp(arr)
    probs = exp / exp.sum()
    return {k: float(p) for k, p in zip(keys, probs)}


# ---------------------------------------------------------------------------
#  SPLNLPBackend
# ---------------------------------------------------------------------------

class SPLNLPBackend:
    """
    Anchor-embedding NLP backend for the Alexandria Semantic Projection Layer.

    For each incoming :class:`~spl.SemanticUnit` the backend computes a
    :class:`~spl.SemanticProjection` whose ``P_r`` distribution is derived
    from cosine similarities between the source text and pre-cached anchor
    embeddings for each relation r ∈ ℛ  (WP2 §3.3).

    Parameters
    ----------
    model_name:
        HuggingFace model identifier for ``SentenceTransformer``.
        Default: ``"all-MiniLM-L6-v2"``  (fast, strong semantic quality).
    builder_origin:
        ``"alpha"`` or ``"beta"`` — dual-builder identity (WP2 §3.3.5).
    matrix_version:
        Relation-matrix version tag written into every projection.
    temperature:
        Softmax temperature *T* ∈ (0, ∞).
        Lower *T* sharpens P_r → favours E1 (singular emission).
        Higher *T* flattens P_r → favours E3 (ambiguity block).
        Default ``0.5`` is calibrated for MiniLM cosine similarity range.
    """

    def __init__(
        self,
        model_name:      str   = "all-MiniLM-L6-v2",
        builder_origin:  str   = "alpha",
        matrix_version:  str   = MATRIX_VERSION,
        temperature:     float = 0.5,
    ) -> None:
        if builder_origin not in ("alpha", "beta"):
            raise ValueError(
                f"builder_origin must be 'alpha' or 'beta', got {builder_origin!r}"
            )
        if temperature <= 0:
            raise ValueError(f"temperature must be > 0, got {temperature}")

        self.builder_origin = builder_origin
        self.matrix_version = matrix_version
        self.temperature    = temperature

        self._model = SentenceTransformer(model_name)

        # Batch-encode all anchor sets once at init-time
        self._rel_embs: dict[str, np.ndarray] = self._encode_anchors(RELATION_ANCHORS)
        self._cat_embs: dict[str, np.ndarray] = self._encode_anchors(CATEGORY_ANCHORS)
        self._mod_embs: dict[str, np.ndarray] = self._encode_anchors(MODALITY_ANCHORS)

    # ------------------------------------------------------------------
    #  Public API
    # ------------------------------------------------------------------

    def create_unit(
        self,
        text:                 str,
        source_ref:           str = "user_input",
        offset_start:         int = 0,
        offset_end:           int = 0,
        fragmentation_signal: str = "",
    ) -> SemanticUnit:
        """Create a :class:`~spl.SemanticUnit` from raw text."""
        return SemanticUnit.new(
            source_text=text,
            source_ref=source_ref,
            offset_start=offset_start,
            offset_end=offset_end or len(text),
            fragmentation_signal=fragmentation_signal,
        )

    def project(self, unit: SemanticUnit) -> SemanticProjection:
        """
        Compute a :class:`~spl.SemanticProjection` for *unit*.

        Embeds ``unit.source_text``, computes cosine similarities against
        every anchor embedding, and applies temperature-scaled softmax to
        obtain normalised probability distributions for P_r, P_category,
        and P_modality.
        """
        emb = self._model.encode([unit.source_text], convert_to_numpy=True)[0]

        P_r        = self._dist(emb, self._rel_embs)
        P_category = self._dist(emb, self._cat_embs)
        P_modality = self._dist(emb, self._mod_embs)

        subjects, objects = self._extract_candidates(unit.source_text)

        return SemanticProjection(
            projection_id=str(uuid.uuid4()),
            unit_id=unit.unit_id,
            builder_origin=self.builder_origin,
            matrix_version=self.matrix_version,
            P_r=P_r,
            subject_candidates=subjects,
            object_candidates=objects,
            P_category=P_category,
            P_modality=P_modality,
            p_illegal=0.0,
        )

    def project_text(
        self,
        text:       str,
        source_ref: str = "user_input",
    ) -> SemanticProjection:
        """Convenience wrapper: :meth:`create_unit` + :meth:`project` in one call."""
        return self.project(self.create_unit(text, source_ref=source_ref))

    # ------------------------------------------------------------------
    #  Internal helpers
    # ------------------------------------------------------------------

    def _encode_anchors(self, anchors: dict[str, str]) -> dict[str, np.ndarray]:
        """Batch-encode anchor phrases; returns ``{relation_key: embedding}``."""
        keys    = list(anchors)
        phrases = [anchors[k] for k in keys]
        embs    = self._model.encode(phrases, convert_to_numpy=True)
        return {k: embs[i] for i, k in enumerate(keys)}

    def _dist(
        self,
        text_emb:    np.ndarray,
        anchor_embs: dict[str, np.ndarray],
    ) -> dict[str, float]:
        """Cosine similarities → temperature-softmax probability distribution."""
        sims = {k: _cosine_sim(text_emb, e) for k, e in anchor_embs.items()}
        return _softmax(sims, self.temperature)

    def _extract_candidates(self, text: str) -> tuple[list[str], list[str]]:
        """
        Extract subject and object candidates from text.

        Tries spaCy dependency parsing first (optional dependency).
        Falls back to :func:`_heuristic_candidates` if spaCy is unavailable
        or yields no results.
        """
        subj, obj = _spacy_candidates(text)
        if subj and obj:
            return subj, obj
        return _heuristic_candidates(text)


# ---------------------------------------------------------------------------
#  Subject / object extraction  (module-level for testability)
# ---------------------------------------------------------------------------

def _spacy_candidates(text: str) -> tuple[list[str], list[str]]:
    """Dependency-based noun-chunk extraction via spaCy (optional)."""
    try:
        import spacy  # type: ignore[import-untyped]

        nlp = spacy.load("en_core_web_sm")
        doc = nlp(text)
        subjects = [
            chunk.text for chunk in doc.noun_chunks
            if chunk.root.dep_ in ("nsubj", "nsubjpass")
        ]
        objects = [
            chunk.text for chunk in doc.noun_chunks
            if chunk.root.dep_ in ("dobj", "pobj", "attr", "dative")
        ]
        return subjects or [], objects or []
    except Exception:
        return [], []


def _heuristic_candidates(text: str) -> tuple[list[str], list[str]]:
    """
    Regex-based fallback: split on a relational verb or preposition.

    Left side  → subject candidates.
    Right side → object candidates.
    If no split point is found the full text is returned on both sides.
    """
    m = _SPLIT_RE.search(text)
    if m:
        left  = text[:m.start()].strip().rstrip(",")
        right = text[m.end():].strip().lstrip(",")
        subj = [left]  if left  else [text]
        obj  = [right] if right else [text]
        return subj, obj
    # No recognisable relational token: use full text as fallback
    return [text], [text]


# ---------------------------------------------------------------------------
#  Dual-builder factory
# ---------------------------------------------------------------------------

def make_dual_backends(
    model_alpha:       str   = "all-MiniLM-L6-v2",
    model_beta:        str   = "all-MiniLM-L6-v2",
    temperature_alpha: float = 0.5,
    temperature_beta:  float = 0.8,
    matrix_version:    str   = MATRIX_VERSION,
) -> tuple[SPLNLPBackend, SPLNLPBackend]:
    """
    Create a matched alpha/beta builder pair for dual-builder workflows (E4).

    Different temperatures produce different P_r sharpness from the same text,
    making non-trivial JSD values observable without different models.
    For maximum independence pass two distinct ``model_alpha``/``model_beta``.

    Returns
    -------
    (alpha_backend, beta_backend)
    """
    alpha = SPLNLPBackend(
        model_alpha,
        builder_origin="alpha",
        matrix_version=matrix_version,
        temperature=temperature_alpha,
    )
    beta = SPLNLPBackend(
        model_beta,
        builder_origin="beta",
        matrix_version=matrix_version,
        temperature=temperature_beta,
    )
    return alpha, beta
