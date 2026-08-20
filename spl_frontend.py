"""Offline reference implementation of SPL modules M1--M3.

This module deliberately has no network, API, or model dependency.  It is a
small, inspectable baseline that turns raw German or English text into the
canonical probabilistic representation consumed by :mod:`spl`.  Its rules are
not presented as universal language understanding: every decision is retained
in ``backend_trace`` and uncertain/unknown input remains uncertain.

Pipeline
--------
M1 ``EpistemicFragmenter``
    text -> minimal relational ``SemanticUnit`` objects
M2 ``DistributionalTypeSystem``
    (surface, context) -> a distribution over the ten WP2 genesis types
M3 ``RelationProjector``
    global family distribution -> typed matrix filter -> relation distribution
    -> sparse R(subject, relation, object) tensor
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
import unicodedata
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Iterable

from spl import SemanticProjection, SemanticUnit, TripleProbability


class EntityType(str, Enum):
    ENTITY = "ENTITY"
    PROPERTY = "PROPERTY"
    PROCESS = "PROCESS"
    EVENT = "EVENT"
    POPULATION = "POPULATION"
    VARIABLE = "VARIABLE"
    MODEL = "MODEL"
    EVIDENCE = "EVIDENCE"
    CLAIM = "CLAIM"
    NORM = "NORM"


class RelationFamily(str, Enum):
    ONTIC = "ONTIC"
    DYNAMIC = "DYNAMIC"
    STATISTICAL = "STATISTICAL"
    EPISTEMIC = "EPISTEMIC"
    MODEL = "MODEL"
    NORMATIVE = "NORMATIVE"


RELATIONS: dict[RelationFamily, tuple[str, ...]] = {
    RelationFamily.ONTIC: (
        "has_property", "part_of", "located_in", "participates_in",
        "instance_of", "capital_of",
    ),
    RelationFamily.DYNAMIC: (
        "affects", "increases", "decreases", "stabilizes", "causes",
        "enables", "inhibits", "triggers",
    ),
    RelationFamily.STATISTICAL: (
        "correlates_with", "covaries_with", "associates_with",
        "differs_from", "varies_across",
    ),
    RelationFamily.EPISTEMIC: (
        "supports", "contradicts", "suggests", "refines", "extends",
        "qualifies", "indicates", "shows",
    ),
    RelationFamily.MODEL: (
        "predicts", "explains", "estimates", "simulates", "approximates",
    ),
    RelationFamily.NORMATIVE: (
        "requires", "forbids", "recommends", "permits", "prioritizes",
    ),
}


@dataclass(frozen=True)
class RelationSignal:
    relation: str
    family: RelationFamily
    start: int
    end: int
    surface: str


@dataclass(frozen=True)
class Extraction:
    subject: str
    object: str
    signals: tuple[RelationSignal, ...]
    scope: str = ""
    modality: str = "asserted"
    temporal: str = "unspecified"


@dataclass(frozen=True)
class TypeDistribution:
    surface: str
    probabilities: dict[str, float]
    h_norm: float
    ambiguous: bool

    @property
    def dominant(self) -> EntityType:
        return EntityType(max(self.probabilities, key=self.probabilities.get))


@dataclass
class CompilationResult:
    source_text: str
    source_ref: str
    language: str
    units: list[SemanticUnit] = field(default_factory=list)
    projections: list[SemanticProjection] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "source_text": self.source_text,
            "source_ref": self.source_ref,
            "language": self.language,
            "units": [unit.to_dict() for unit in self.units],
            "projections": [projection.to_dict() for projection in self.projections],
        }


# Ordered longest/specific first.  Relation IDs are language-independent.
_CUES: tuple[tuple[str, RelationFamily, str], ...] = (
    ("capital_of", RelationFamily.ONTIC,
     r"\b(?:is|was)\s+(?:the\s+)?capital\s+of\b|\b(?:ist|war)\s+(?:die\s+)?hauptstadt\s+von\b"),
    ("correlates_with", RelationFamily.STATISTICAL,
     r"\bcorrelat(?:es|ed|e|ing)\s+with\b|\bkorrelier(?:t|te|en)\s+mit\b"),
    ("associates_with", RelationFamily.STATISTICAL,
     r"\b(?:is|was)\s+associated\s+with\b|\b(?:ist|war)\s+(?:mit\s+)?assoziiert\s+mit\b"),
    ("varies_across", RelationFamily.STATISTICAL,
     r"\bvar(?:y|ies|ied)\s+across\b|\bvariier(?:t|te|en)\s+(?:über|zwischen)\b"),
    ("suggests", RelationFamily.EPISTEMIC,
     r"\bsuggest(?:s|ed)?\s+(?:that\s+)?|\b(?:legt|legen|legte)\s+nahe[,]?\s*(?:dass\s+)?|\bdeutet\s+darauf\s+hin[,]?\s*dass\s+"),
    ("indicates", RelationFamily.EPISTEMIC,
     r"\bindicat(?:e|es|ed)\s+(?:that\s+)?|\b(?:zeigt|zeigen|zeigte)\s+(?:an[,]?\s*)?(?:dass\s+)?"),
    ("causes", RelationFamily.DYNAMIC,
     r"\bcaus(?:e|es|ed|ing)\b|\bverursach(?:t|en|te)\b"),
    ("increases", RelationFamily.DYNAMIC,
     r"\bincreas(?:e|es|ed|ing)\b|\berhöh(?:t|en|te)\b|\bsteigert\b"),
    ("decreases", RelationFamily.DYNAMIC,
     r"\b(?:decreas(?:e|es|ed|ing)|reduc(?:e|es|ed|ing))\b|\b(?:reduzier|verringer)(?:t|en|te)\b|\bsenkt\b"),
    ("inhibits", RelationFamily.DYNAMIC,
     r"\binhibit(?:s|ed|ing)?\b|\bhemm(?:t|en|te)\b"),
    ("enables", RelationFamily.DYNAMIC,
     r"\benabl(?:e|es|ed|ing)\b|\bermöglich(?:t|en|te)\b"),
    ("affects", RelationFamily.DYNAMIC,
     r"\baffect(?:s|ed|ing)?\b|\bbeeinfluss(?:t|en|te)\b"),
    ("supports", RelationFamily.EPISTEMIC,
     r"\bsupport(?:s|ed|ing)?\b|\bunterstütz(?:t|en|te)\b"),
    ("contradicts", RelationFamily.EPISTEMIC,
     r"\bcontradict(?:s|ed|ing)?\b|\bwiderspr(?:icht|echen|ach)\b"),
    ("predicts", RelationFamily.MODEL,
     r"\bpredict(?:s|ed|ing)?\b|\bprognostizier(?:t|en|te)\b"),
    ("requires", RelationFamily.NORMATIVE,
     r"\brequir(?:e|es|ed|ing)\b|\berforder(?:t|n|te)\b"),
    ("recommends", RelationFamily.NORMATIVE,
     r"\brecommend(?:s|ed|ing)?\b|\bempf(?:iehlt|ehlen|ahl)\b"),
    ("located_in", RelationFamily.ONTIC,
     r"\b(?:is|are|was|were)\s+located\s+in\b|\b(?:liegt|liegen|lag)\s+in\b"),
    ("has_property", RelationFamily.ONTIC,
     r"\b(?:has|have|had)\b|\b(?:hat|haben|hatte)\b"),
)

_COMPLEMENT = re.compile(
    r"\b(?:suggest(?:s|ed)?\s+that|indicat(?:e|es|ed)\s+that|shows?\s+that|"
    r"legt\s+nahe[,]?\s+dass|zeigt\s+(?:an[,]?\s*)?dass|deutet\s+darauf\s+hin[,]?\s+dass)\s+",
    re.IGNORECASE,
)
_CONTRAST = re.compile(r"\s*,?\s*\b(?:although|whereas|however|obwohl|wohingegen|jedoch)\b\s*", re.IGNORECASE)
_CONJUNCTION = re.compile(r"\s*,?\s+\b(?:and|und)\b\s+", re.IGNORECASE)
_SCOPE = re.compile(
    r"\b(under certain conditions|in some sectors|across sectors|"
    r"unter bestimmten bedingungen|in einigen sektoren|bereichsweise)\b",
    re.IGNORECASE,
)


def detect_relation_signals(text: str) -> list[RelationSignal]:
    found: list[RelationSignal] = []
    for relation, family, pattern in _CUES:
        for match in re.finditer(pattern, text, re.IGNORECASE):
            found.append(RelationSignal(relation, family, match.start(), match.end(), match.group(0).strip()))
    found.sort(key=lambda signal: (signal.start, -(signal.end - signal.start)))

    # An epistemic complement contains another proposition.  M1 emits that
    # proposition separately, so the outer unit keeps only its first cue.
    if found and found[0].family == RelationFamily.EPISTEMIC and _COMPLEMENT.search(text):
        return [found[0]]

    # Multiple cues belong to one unit only when coordinated directly.
    if len(found) > 1:
        coordinated = [found[0]]
        for signal in found[1:]:
            between = text[coordinated[-1].end:signal.start]
            if re.fullmatch(r"\s*(?:,|or|oder|and|und)?\s*", between, re.IGNORECASE):
                coordinated.append(signal)
            else:
                break
        return coordinated
    return found


class EpistemicFragmenter:
    """M1: boundary-driven, offset-preserving epistemic fragmentation."""

    def fragment(self, text: str, source_ref: str = "", language: str = "auto") -> list[SemanticUnit]:
        lang = detect_language(text) if language == "auto" else language
        spans: list[tuple[int, int, str]] = []
        for sentence in re.finditer(r"[^.!?]+(?:[.!?]+|$)", text):
            raw = sentence.group(0)
            left = len(raw) - len(raw.lstrip())
            right = len(raw.rstrip(" \t\r\n.!?"))
            if right <= left:
                continue
            start = sentence.start() + left
            end = sentence.start() + right
            spans.extend(self._decompose(text[start:end], start))

        unique: list[tuple[int, int, str]] = []
        seen: set[tuple[int, int]] = set()
        for start, end, signal in sorted(spans):
            if (start, end) in seen:
                continue
            seen.add((start, end))
            unique.append((start, end, signal))

        units = []
        for start, end, signal in unique:
            fragment = text[start:end].strip(" ,;:\t\r\n")
            adjustment = len(text[start:end]) - len(text[start:end].lstrip(" ,;:\t\r\n"))
            start += adjustment
            end = start + len(fragment)
            if not fragment or not detect_relation_signals(fragment):
                continue
            context_start = max(0, start - 120)
            context_end = min(len(text), end + 120)
            unit = SemanticUnit.new(
                fragment,
                source_ref,
                start,
                end,
                signal,
                source_language=lang,
                context_window=text[context_start:context_end],
            )
            units.append(unit)
        return units

    def _decompose(self, clause: str, base: int) -> list[tuple[int, int, str]]:
        contrast = _CONTRAST.search(clause)
        if contrast:
            left = clause[:contrast.start()].rstrip(" ,")
            right = clause[contrast.end():].lstrip(" ,")
            right_start = base + contrast.end() + (len(clause[contrast.end():]) - len(right))
            return self._decompose(left, base) + self._decompose(right, right_start)

        for conjunction in _CONJUNCTION.finditer(clause):
            left = clause[:conjunction.start()].rstrip(" ,")
            right = clause[conjunction.end():].lstrip(" ,")
            if detect_relation_signals(left) and detect_relation_signals(right):
                right_start = base + conjunction.end() + (len(clause[conjunction.end():]) - len(right))
                return self._decompose(left, base) + self._decompose(right, right_start)

        result = [(base, base + len(clause), "relational_cue")]
        complement = _COMPLEMENT.search(clause)
        if complement:
            inner_raw = clause[complement.end():]
            inner = inner_raw.lstrip(" ,")
            inner_start = base + complement.end() + (len(inner_raw) - len(inner))
            result.extend(self._decompose(inner, inner_start))
        return result


class DistributionalTypeSystem:
    """M2: an inspectable distributional baseline over WP2's T_v1.0."""

    _LEXICON: dict[EntityType, set[str]] = {
        EntityType.PROPERTY: {"productivity", "produktivität", "effect", "effekt", "color", "farbe", "size", "größe"},
        EntityType.PROCESS: {"work", "arbeit", "activity", "aktivität", "growth", "wachstum", "production", "produktion", "process", "prozess"},
        EntityType.EVENT: {"election", "wahl", "accident", "unfall", "war", "krieg", "meeting", "treffen"},
        EntityType.POPULATION: {"workers", "arbeitnehmer", "patients", "patienten", "people", "menschen", "sectors", "sektoren"},
        EntityType.VARIABLE: {"temperature", "temperatur", "pressure", "druck", "concentration", "konzentration", "rate", "quote"},
        EntityType.MODEL: {"model", "modell", "simulation", "forecast", "prognose", "algorithm", "algorithmus"},
        EntityType.EVIDENCE: {"evidence", "beleg", "results", "ergebnisse", "study", "studie", "data", "daten", "observation", "beobachtung"},
        EntityType.CLAIM: {"claim", "behauptung", "hypothesis", "hypothese", "proposition"},
        EntityType.NORM: {"rule", "regel", "law", "gesetz", "guideline", "richtlinie", "norm", "regulation", "verordnung"},
    }

    def classify(self, surface: str, context: str = "", role: str = "") -> TypeDistribution:
        scores = {kind: 0.02 for kind in EntityType}
        clean = _normal_form(surface)
        words = set(clean.split())
        scores[EntityType.ENTITY] += 0.18

        for kind, vocabulary in self._LEXICON.items():
            if words & vocabulary or clean in vocabulary:
                scores[kind] += 0.82

        if surface.strip().startswith("[claim:"):
            scores[EntityType.CLAIM] += 0.90
        if role == "subject" and re.search(r"\b(suggest|indicat|show|zeig|nahe)\w*\b", context, re.IGNORECASE):
            scores[EntityType.EVIDENCE] += 0.35
        if surface[:1].isupper() and not words & {item for values in self._LEXICON.values() for item in values}:
            scores[EntityType.ENTITY] += 0.55
        if re.search(r"\b(variable|variable|wert|value)\b", context, re.IGNORECASE):
            scores[EntityType.VARIABLE] += 0.25

        probabilities = _normalise({kind.value: value for kind, value in scores.items()})
        entropy = _h_norm(probabilities)
        return TypeDistribution(surface, probabilities, entropy, entropy >= 0.72)


class TypedRelationMatrix:
    """Versioned M_v1.0 baseline. Missing pairs are intentionally illegal."""

    version = "M_v1.0-offline-reference"

    def __init__(self) -> None:
        D = RelationFamily.DYNAMIC
        S = RelationFamily.STATISTICAL
        O = RelationFamily.ONTIC
        E = RelationFamily.EPISTEMIC
        M = RelationFamily.MODEL
        N = RelationFamily.NORMATIVE
        T = EntityType
        self.rules: dict[tuple[EntityType, EntityType], frozenset[RelationFamily]] = {
            (T.ENTITY, T.ENTITY): frozenset({O, D, S}),
            (T.ENTITY, T.PROPERTY): frozenset({O, D}),
            (T.ENTITY, T.PROCESS): frozenset({O, D}),
            (T.PROCESS, T.PROPERTY): frozenset({D, S}),
            (T.PROCESS, T.PROCESS): frozenset({D}),
            (T.PROPERTY, T.POPULATION): frozenset({S}),
            (T.EVIDENCE, T.CLAIM): frozenset({E}),
            (T.EVIDENCE, T.EVIDENCE): frozenset({E, S}),
            (T.MODEL, T.EVENT): frozenset({M}),
            (T.MODEL, T.PROPERTY): frozenset({M}),
            (T.MODEL, T.VARIABLE): frozenset({M}),
            (T.NORM, T.PROCESS): frozenset({N}),
            (T.NORM, T.ENTITY): frozenset({N}),
            (T.VARIABLE, T.VARIABLE): frozenset({S, D}),
        }
        canonical = [
            (subject.value, object_.value, sorted(family.value for family in families))
            for (subject, object_), families in sorted(
                self.rules.items(), key=lambda item: (item[0][0].value, item[0][1].value)
            )
        ]
        self.seal_hash = hashlib.sha256(
            json.dumps(canonical, separators=(",", ":")).encode("utf-8")
        ).hexdigest()

    def permitted(self, subject: EntityType, object_: EntityType) -> frozenset[RelationFamily]:
        return self.rules.get((subject, object_), frozenset())

    def to_dict(self) -> dict:
        return {
            "@context": {"alexandria": "https://protocol.alexandria.org/v2.2/vocab#"},
            "@type": "alexandria:TypedRelationMatrix",
            "version": self.version,
            "seal_hash": self.seal_hash,
            "rules": [
                {
                    "subjectType": subject.value,
                    "objectType": object_.value,
                    "permittedFamilies": sorted(family.value for family in families),
                }
                for (subject, object_), families in sorted(
                    self.rules.items(), key=lambda item: (item[0][0].value, item[0][1].value)
                )
            ],
        }


class RelationProjector:
    """M3: two-stage global-family and typed local-relation projection."""

    def __init__(
        self,
        type_system: DistributionalTypeSystem | None = None,
        matrix: TypedRelationMatrix | None = None,
    ) -> None:
        self.type_system = type_system or DistributionalTypeSystem()
        self.matrix = matrix or TypedRelationMatrix()

    def project(self, unit: SemanticUnit, builder_origin: str = "alpha") -> SemanticProjection:
        extraction = extract(unit.source_text)
        if extraction is None:
            return SemanticProjection(
                projection_id=str(uuid.uuid4()), unit_id=unit.unit_id,
                builder_origin=builder_origin, matrix_version=self.matrix.version,
                P_r={}, subject_candidates=[], object_candidates=[],
                source_ref=unit.source_ref, matrix_seal_hash=self.matrix.seal_hash,
                uncertainty={"parse_failure": True},
                backend_trace={"backend": "offline-rule-reference-v1", "signals": []},
            )

        subject_type = self.type_system.classify(extraction.subject, unit.context_window or unit.source_text, "subject")
        object_surface = extraction.object
        if extraction.signals[0].family == RelationFamily.EPISTEMIC and _COMPLEMENT.search(unit.source_text):
            object_surface = f"[claim: {object_surface}]"
        object_type = self.type_system.classify(object_surface, unit.context_window or unit.source_text, "object")
        permitted = self.matrix.permitted(subject_type.dominant, object_type.dominant)

        P_family = self._family_distribution(extraction.signals)
        p_illegal = sum(probability for family, probability in P_family.items() if RelationFamily(family) not in permitted)
        legal_signals = [signal for signal in extraction.signals if signal.family in permitted]
        P_r = self._relation_distribution(legal_signals)

        subject_id = canonical_entity_id(extraction.subject)
        object_id = canonical_entity_id(object_surface)
        triples = [
            TripleProbability(
                subject=extraction.subject,
                relation=relation,
                object=object_surface,
                probability=probability,
                subject_id=subject_id,
                object_id=object_id,
                subject_type=subject_type.dominant.value,
                object_type=object_type.dominant.value,
            )
            for relation, probability in P_r.items()
        ]
        category = self._category_distribution(extraction.signals)
        modality = {extraction.modality: 0.95, "asserted": 0.05} if extraction.modality != "asserted" else {"asserted": 1.0}
        scope = {extraction.scope: 1.0} if extraction.scope else {}
        temporal = {extraction.temporal: 1.0}

        return SemanticProjection(
            projection_id=str(uuid.uuid4()),
            unit_id=unit.unit_id,
            builder_origin=builder_origin,
            matrix_version=self.matrix.version,
            P_r=P_r,
            subject_candidates=[extraction.subject],
            object_candidates=[object_surface],
            P_category=category,
            P_modality=modality,
            P_scope=scope,
            triple_distribution=triples,
            P_subject={extraction.subject: 1.0},
            P_object={object_surface: 1.0},
            P_family=P_family,
            P_temporal=temporal,
            entity_type_distributions={
                "subject": subject_type.probabilities,
                "object": object_type.probabilities,
            },
            uncertainty={
                "subject_type_h_norm": subject_type.h_norm,
                "object_type_h_norm": object_type.h_norm,
                "ambiguous_type": subject_type.ambiguous or object_type.ambiguous,
            },
            source_ref=unit.source_ref,
            p_illegal=p_illegal,
            matrix_seal_hash=self.matrix.seal_hash,
            backend_trace={
                "backend": "offline-rule-reference-v1",
                "signals": [
                    {"surface": signal.surface, "relation": signal.relation, "family": signal.family.value}
                    for signal in extraction.signals
                ],
                "dominant_types": [subject_type.dominant.value, object_type.dominant.value],
                "permitted_families": sorted(family.value for family in permitted),
            },
        )

    @staticmethod
    def _family_distribution(signals: Iterable[RelationSignal]) -> dict[str, float]:
        signals = list(signals)
        counts = {family: 0 for family in RelationFamily}
        for signal in signals:
            counts[signal.family] += 1
        active = [family for family, count in counts.items() if count]
        if not active:
            return {family.value: 1.0 / len(RelationFamily) for family in RelationFamily}
        inactive = [family for family in RelationFamily if family not in active]
        result = {family.value: 0.94 * counts[family] / len(signals) for family in active}
        for family in inactive:
            result[family.value] = 0.06 / len(inactive) if inactive else 0.0
        return _normalise(result)

    @staticmethod
    def _relation_distribution(signals: list[RelationSignal]) -> dict[str, float]:
        if not signals:
            return {}
        unique: list[RelationSignal] = []
        seen: set[str] = set()
        for signal in signals:
            if signal.relation not in seen:
                unique.append(signal)
                seen.add(signal.relation)
        if len(unique) > 1:
            return {signal.relation: 1.0 / len(unique) for signal in unique}

        signal = unique[0]
        alternatives = [relation for relation in RELATIONS[signal.family] if relation != signal.relation]
        if not alternatives:
            return {signal.relation: 1.0}
        return {signal.relation: 0.97, alternatives[0]: 0.03}

    @staticmethod
    def _category_distribution(signals: tuple[RelationSignal, ...]) -> dict[str, float]:
        mapping = {
            RelationFamily.ONTIC: "ontic",
            RelationFamily.DYNAMIC: "dynamic",
            RelationFamily.STATISTICAL: "statistical",
            RelationFamily.EPISTEMIC: "epistemic",
            RelationFamily.MODEL: "model",
            RelationFamily.NORMATIVE: "normative",
        }
        counts: dict[str, float] = {}
        for signal in signals:
            key = mapping[signal.family]
            counts[key] = counts.get(key, 0.0) + 1.0
        return _normalise(counts)


class SemanticCompiler:
    """Convenience facade that runs raw text through M1, M2, and M3."""

    def __init__(
        self,
        fragmenter: EpistemicFragmenter | None = None,
        projector: RelationProjector | None = None,
    ) -> None:
        self.fragmenter = fragmenter or EpistemicFragmenter()
        self.projector = projector or RelationProjector()

    def compile(
        self,
        text: str,
        source_ref: str = "",
        language: str = "auto",
        builder_origin: str = "alpha",
    ) -> CompilationResult:
        resolved_language = detect_language(text) if language == "auto" else language
        units = self.fragmenter.fragment(text, source_ref, resolved_language)
        projections = [self.projector.project(unit, builder_origin) for unit in units]
        return CompilationResult(text, source_ref, resolved_language, units, projections)


def extract(text: str) -> Extraction | None:
    signals = detect_relation_signals(text)
    if not signals:
        return None
    first, last = signals[0], signals[-1]
    subject = _clean_subject(text[:first.start])
    object_ = _clean_object(text[last.end:])
    if not subject or not object_:
        return None
    scope_match = _SCOPE.search(object_)
    scope = scope_match.group(1) if scope_match else ""
    if scope_match:
        object_ = (object_[:scope_match.start()] + object_[scope_match.end():]).strip(" ,;")
    modality = "possible" if re.search(r"\b(?:may|might|could|possibly|kann|könnte|möglicherweise)\b", text, re.IGNORECASE) else "asserted"
    if signals[0].family == RelationFamily.EPISTEMIC:
        modality = "suggested"
    temporal = _temporal(text)
    return Extraction(subject, object_, tuple(signals), scope, modality, temporal)


def detect_language(text: str) -> str:
    lowered = f" {_normal_form(text)} "
    german = sum(token in lowered for token in (" der ", " die ", " das ", " und ", " ist ", " von ", " mit ", " dass "))
    english = sum(token in lowered for token in (" the ", " and ", " is ", " of ", " with ", " that ", " may "))
    return "de" if german > english else "en"


_ENTITY_ALIASES = {
    "paris": "geo:paris",
    "france": "geo:france",
    "frankreich": "geo:france",
    "remote work": "concept:remote_work",
    "remote-arbeit": "concept:remote_work",
    "fernarbeit": "concept:remote_work",
    "productivity": "concept:productivity",
    "produktivität": "concept:productivity",
}


def canonical_entity_id(surface: str) -> str:
    clean = _normal_form(surface.strip("[]"))
    if clean.startswith("claim:"):
        digest = hashlib.sha256(clean[6:].strip().encode("utf-8")).hexdigest()[:20]
        return f"claim:{digest}"
    if clean in _ENTITY_ALIASES:
        return _ENTITY_ALIASES[clean]
    slug = re.sub(r"[^a-z0-9]+", "_", clean).strip("_")
    return f"concept:{slug}" if slug else "concept:unknown"


def _clean_subject(value: str) -> str:
    value = value.strip(" ,;:")
    value = re.sub(r"^(?:the|a|an|der|die|das|ein|eine)\s+", "", value, flags=re.IGNORECASE)
    value = re.sub(r"\s+\b(?:may|might|could|can|kann|könnte)\b\s*$", "", value, flags=re.IGNORECASE)
    return value.strip()


def _clean_object(value: str) -> str:
    value = value.strip(" ,;:")
    value = re.sub(r"^(?:or|oder|and|und|that|dass)\s+", "", value, flags=re.IGNORECASE)
    value = re.sub(r"^(?:the|a|an|der|die|das|den|dem|ein|eine)\s+", "", value, flags=re.IGNORECASE)
    return value.strip()


def _temporal(text: str) -> str:
    lowered = _normal_form(text)
    checks = (
        (r"\b(before|vor)\b", "before"),
        (r"\b(after|nach)\b", "after"),
        (r"\b(during|während)\b", "during"),
        (r"\b(overlaps?|überlappt)\b", "overlaps"),
    )
    for pattern, value in checks:
        if re.search(pattern, lowered):
            return value
    return "unspecified"


def _normal_form(value: str) -> str:
    return unicodedata.normalize("NFKC", value).casefold().strip()


def _normalise(values: dict[str, float]) -> dict[str, float]:
    total = sum(max(0.0, value) for value in values.values())
    if total <= 0:
        return {}
    return {key: max(0.0, value) / total for key, value in values.items()}


def _h_norm(distribution: dict[str, float]) -> float:
    if len(distribution) <= 1:
        return 0.0
    entropy = -sum(probability * math.log2(probability) for probability in distribution.values() if probability > 0)
    return entropy / math.log2(len(distribution))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Compile raw text through Alexandria SPL M1--M3")
    parser.add_argument("text", nargs="?", help="German or English source text; stdin when omitted")
    parser.add_argument("--source-ref", default="cli:stdin")
    parser.add_argument("--language", choices=("auto", "de", "en"), default="auto")
    arguments = parser.parse_args(argv)
    source_text = arguments.text if arguments.text is not None else sys.stdin.read()
    result = SemanticCompiler().compile(source_text, arguments.source_ref, arguments.language)
    print(json.dumps(result.to_dict(), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
