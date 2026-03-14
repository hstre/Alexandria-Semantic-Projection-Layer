# Alexandria — Semantic Projection Layer

[![CI](https://github.com/hstre/Alexandria-Semantic-Projection-Layer/actions/workflows/ci.yml/badge.svg)](https://github.com/hstre/Alexandria-Semantic-Projection-Layer/actions/workflows/ci.yml)

**Working Paper 2 · Formal Bridge between Natural Language and Epistemic Protocol**

> *The SPL transforms natural language segments into probabilistic relational tensors from which canonical claim candidates are derived through entropy-constrained projection rules.*

**Hanns-Steffen Rentschler · 2026**  
Part of the [Alexandria Protocol](https://github.com/hstre/Alexandria-Protokoll) ecosystem.

---

## Three-Layer Architecture

```
┌────────────────────────────────────────────────────────────┐
│  spl.py  —  PROBABILISTIC PRE-PROTOCOL STAGE               │
│  ────────────────────────────────────────────────────────  │
│  Text → SemanticUnit → SemanticProjection → ClaimCandidate │
│  Operates on distributions P_r over relation space         │
│  Quantifies ambiguity (H_norm) and divergence (JSD)        │
└─────────────────────────────┬──────────────────────────────┘
                              │  ClaimCandidates (probabilistic)
               ┌──────────────▼──────────────┐
               │  spl_gateway.py             │
               │  LEGAL PROTOCOL ENTRY POINT │
               │  ─────────────────────────  │
               │  validate_candidate_for_    │
               │    protocol_entry()         │
               │  ClaimCandidateConverter    │
               │  emit_claim_nodes()  ← THE  │
               │    ONLY legal path to       │
               │    ClaimNode                │
               │                             │
               │  Validates: emission rule,  │
               │  confidence, entropy, JSD,  │
               │  evidence count             │
               │  Assigns: SHA256 claim_id             │
               │  Logs: GatewayEvent → audit_log.json  │
               └───────────┬───────────┘
                           │
┌──────────────────────────▼──────────────────────────┐
│  Protocol Layer (schema.py)   DETERMINISTIC         │
│  ─────────────────────────────────────────────────  │
│  ClaimNode → ClaimGraph                             │
│  Diff, Adjudication, Branch, Seal                   │
│  Operates on discrete, sealed epistemic objects     │
└─────────────────────────────────────────────────────┘
```

**SPL is probabilistic.** It operates on distributions over relation spaces and
quantifies ambiguity and builder divergence mathematically.

**The Protocol is deterministic.** It operates on discrete, sealed claim objects
with no distributional uncertainty.

**The Gateway is the boundary.** It translates from probabilistic to deterministic
by validating each candidate against threshold criteria and assigning a
deterministic SHA256 identity to every emitted ClaimNode. Nothing enters the
ClaimGraph without passing through the gateway.

---

## What is the SPL?

The Alexandria Protocol operates on discrete, structured claim objects. But claims originate from natural language. The **Semantic Projection Layer (SPL)** is the formally defined pre-protocol stage that bridges this gap.

Without the SPL, direct text-to-claim mapping produces:
- **Unstable claims** — small phrasing variations create artificial diffs
- **Language-dependent artifacts** — same content in different languages yields different structures
- **Unrepresented ambiguity** — epistemic hedging is discarded instead of preserved

The SPL solves this by introducing a probabilistic intermediate representation.

---

## Core Operations

```
Source Text
    │
    ▼  Fragmentation                        spl.py
SemanticUnit          minimal epistemic unit (subject · relation · object)
    │
    ▼  Projection                           spl.py
SemanticProjection    probability tensor over relation space
    │
    ▼  Emission (E0–E4)                     spl.py
ClaimCandidate        structured, scored, entropy-annotated
    │
    ▼  SPLGateway.emit_claim_nodes()        spl_gateway/  ← BOUNDARY
       uses ClaimCandidateConverter internally;
       validates, converts, hashes, logs
ClaimNode             Alexandria canonical claim
```

**Protocol invariant [SHALL]:** No text fragment may become a ClaimNode directly.
The boundary is `spl_gateway` — specifically `SPLGateway.emit_claim_nodes()`.
`ClaimCandidateConverter` is the conversion mechanism *used by* the gateway;
it does not define the boundary.

---

## Key Concepts

| Concept | Description |
|---------|-------------|
| `SemanticUnit` | Minimal epistemic fragment: (subject, relation_candidates, object) |
| `SemanticProjection` | Probability distribution over typed relation space |
| `MappingConfidence` | MAPPED / CANDIDATE / LOW_CONFIDENCE / MULTIPLE_CANDIDATES / UNMAPPED |
| Shannon Entropy H | Ambiguity measure — high H → emission rule E3 (AMBIGUOUS) |
| Jensen-Shannon Divergence | Builder divergence measure — JSD > τ₄ → BRANCH_CANDIDATE |
| `ClaimCandidate` | Scored, typed candidate with projection metadata |

## Emission Rules

| Rule | Trigger | Output |
|------|---------|--------|
| E0 | Structural rejection (no valid type-pair) | Rejected |
| E1 | Single dominant relation (p > τ₁) | Single ClaimCandidate |
| E2 | Top-k above threshold | k ClaimCandidates |
| E3 | High entropy (H > τ₃) | AMBIGUOUS — human review |
| E4 | Builder divergence (JSD > τ₄) | BRANCH_CANDIDATE |

## Thresholds Θ

| Parameter | Value | Meaning |
|-----------|-------|---------|
| τ₀ | 0.50 | Minimum projection mass for any candidate |
| τ₁ | 0.60 | Singular dominance threshold |
| τ₂ | 0.25 | Minimum mass for top-k inclusion |
| τ₃ | 0.65 | Entropy threshold → AMBIGUOUS |
| τ₄ | 0.40 | JSD threshold → BRANCH_CANDIDATE |

---

## Repository Contents

```
nlp_backend.py                      Anchor-embedding NLP backend.
                                    Computes P_r, P_category, P_modality
                                    via sentence-transformers cosine similarity
                                    against versioned anchor phrases (ℛ v2.2.0-SML).
                                    Includes make_dual_backends() for E4 workflows.

spl.py                              Probabilistic pre-protocol stage:
                                    SemanticUnit, SemanticProjection,
                                    EmissionEngine E0–E4, SPLThresholds,
                                    compute_h_norm, compute_jsd

spl_gateway/                        THE BOUNDARY — legal protocol entry point.
  __init__.py                       Public API re-export (no import changes needed)
  _exceptions.py                    SPLGatewayError, CandidateRejectedError,
                                    ClaimValidationError
  _types.py                         GatewayEvent, SPLResult, DualBuilderResult
  _utils.py                         canonicalize_*, hash_claim, validate_claim_node
  _converter.py                     ClaimCandidateConverter — conversion mechanism
                                    used BY the gateway; not the boundary itself.
                                    Also: _CATEGORY_HINT_MAP, _MODALITY_HINT_MAP,
                                    validate_candidate_for_protocol_entry()
  _gateway.py                       SPLGateway — enforces the boundary via
                                    emit_claim_nodes() (the ONLY legal path to
                                    ClaimNode). make_gateway() factory.
WP2_Semantic_Projection_Layer.md    Full working paper (theory)

tests/
  test_entropy.py                   H_norm unit tests
  test_jsd.py                       JSD unit tests
  test_spl_rules.py                 Emission rules E0–E4 + end-to-end pipeline
  test_gateway.py                   Gateway boundary unit tests (97 tests total)
  test_nlp_backend.py               NLP backend unit tests (mocked model)

examples/
  demo_pipeline.py                  Pipeline walkthrough — all four emission
                                    rules (E1–E4) with annotated output.
                                    Run: python examples/demo_pipeline.py
  theory_verification.py            Full WP2 theory verification — exhaustive
                                    E0–E4 tests with verbose annotation.
                                    Run: python examples/theory_verification.py
  simple_claim.txt                  "Paris is the capital of France." (E1)
  ambiguous_claim.txt               Modal+conjunctive hedging → E3 block
  multi_claim.txt                   3-unit sentence → mixed E1/E2 output

requirements.txt                    sentence-transformers, numpy

audit_log.json                      GatewayEvent log (auto-generated at runtime)
README.md                           This file
```

The reference implementation (`spl.py`) is also part of `hstre/Alexandria-Protokoll` as `alexandria_core/spl.py`.

---

## NLP Backend

The `nlp_backend.py` module implements real semantic projection via anchor embeddings.
It translates raw text into a `SemanticProjection` without any manually specified distribution.

### Installation

```bash
pip install -r requirements.txt
```

`requirements.txt` covers all runtime dependencies:

```
sentence-transformers>=2.7.0
numpy>=1.24.0
```

The default model (`all-MiniLM-L6-v2`, ~80 MB) is downloaded automatically from HuggingFace
on first use. No GPU required. No network access needed after the first run.

spaCy is an **optional** dependency. If available, it improves subject/object extraction via
dependency parsing. Without it the backend falls back to regex heuristics automatically.

### Quick start

```python
from nlp_backend import SPLNLPBackend

backend = SPLNLPBackend()                                 # downloads model once
proj    = backend.project_text("Paris is the capital of France.")

print(proj.P_r)          # {"capital_of": 0.68, "is_a": 0.11, ...}
print(proj.P_category)   # {"ontic": 0.62, ...}
print(proj.P_modality)   # {"asserted": 0.91, ...}
```

### Full pipeline

```python
from nlp_backend import SPLNLPBackend
from spl import EmissionEngine
from spl_gateway import SPLGateway

backend = SPLNLPBackend()
engine  = EmissionEngine()
gateway = SPLGateway()

proj       = backend.project_text("Paris is the capital of France.")
candidates = engine.emit(proj)                            # E1 / E2 / E3 / E0
nodes      = gateway.emit_claim_nodes(candidates)         # ClaimNode list

for node in nodes:
    print(node)   # ClaimNode('Paris' --[capital_of]--> 'France')
```

### Dual-builder (E4 divergence detection)

```python
from nlp_backend import make_dual_backends
from spl_gateway import SPLGateway

alpha, beta = make_dual_backends()          # T_alpha=0.5, T_beta=0.8
gateway     = SPLGateway()
text        = "Aspirin may reduce cardiovascular risk."

dual = gateway.submit_dual(
    alpha.project_text(text),
    beta.project_text(text),
)
if dual.branched:
    print(f"Builder divergence JSD={dual.jsd:.3f} → BRANCH_CANDIDATE")
```

### Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `model_name` | `"all-MiniLM-L6-v2"` | HuggingFace SentenceTransformer model |
| `builder_origin` | `"alpha"` | `"alpha"` or `"beta"` |
| `temperature` | `0.5` | Softmax sharpness (lower → sharper P_r) |
| `matrix_version` | `"v2.2.0-SML"` | Relation matrix version tag (do not change) |

---

## Status

This is a **working paper** — the theory is stable, the implementation is complete, the NLP backend is operational.

Open items:
- τ₂ calibration against gold-standard corpora
- Evaluation against benchmark plan (Section 4 / WP2 Appendix I)

---

## Related

- [Alexandria Protocol](https://github.com/hstre/Alexandria-Protokoll) — the core protocol this paper extends
- SSRN submission pending

## License

Paper: CC BY 4.0  
Code: MIT

© 2026 Hanns-Steffen Rentschler
