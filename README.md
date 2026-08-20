# Alexandria — Semantic Projection Layer

**Working Paper 2 · Formal Bridge between Natural Language and Epistemic Protocol**

> *The SPL transforms natural language segments into probabilistic relational tensors from which canonical claim candidates are derived through entropy-constrained projection rules.*

**Hanns-Steffen Rentschler · 2026**  
Part of the [Alexandria Protocol](https://github.com/hstre/Alexandria-Protokoll) ecosystem.

---

## Three-Layer Architecture

```
┌─────────────────────────────────────────────────────┐
│  SPL Layer (spl_frontend.py + spl.py) PROBABILISTIC │
│  ─────────────────────────────────────────────────  │
│  Text → SemanticUnit → SemanticProjection           │
│  Operates on distributions P_r over relation space  │
│  Quantifies ambiguity (H_norm) and divergence (JSD) │
└──────────────────────────┬──────────────────────────┘
                           │
               ┌───────────▼───────────┐
               │  Gateway (spl_gateway.py)             │
               │  ─────────────────────────────────── │
               │  emit_claim_nodes()  ← only legal     │
               │  path to ClaimNode                    │
               │                                       │
               │  Validates: emission rule, confidence, │
               │  entropy, JSD, evidence count         │
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
    ▼  Fragmentation
SemanticUnit          minimal epistemic unit (subject · relation · object)
    │
    ▼  Projection
SemanticProjection    probability tensor over relation space
    │
    ▼  Emission (E0–E4)
ClaimCandidate        structured, scored, entropy-annotated
    │
    ▼  ClaimCandidateConverter  ← protocol boundary
ClaimNode             Alexandria canonical claim
```

**Protocol invariant [SHALL]:** No text fragment may become a ClaimNode directly. The path above is the only legal entry into the Alexandria graph.

## Offline M1--M3 reference pipeline

The repository includes an inspectable, deterministic German/English baseline.
It has no network, API-key, hosted-model, or third-party package dependency:

```python
from spl_frontend import SemanticCompiler
from spl import EmissionEngine

result = SemanticCompiler().compile(
    "Paris ist die Hauptstadt von Frankreich.",
    source_ref="document:1",
)
projection = result.projections[0]
candidate = EmissionEngine().emit(projection)[0]

assert candidate.relation == "capital_of"
assert candidate.object_id == "geo:france"
```

The same representation is available as JSON from the command line:

```bash
python spl_frontend.py "Paris is the capital of France."
```

Every heuristic decision is exposed in `backend_trace`. Unknown constructions
are not completed by a hidden language model; they remain parse failures,
ambiguous projections, or structural violations.

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
| τ₀ | 0.50 | Structural rejection threshold for illegal family mass |
| τ₁ | 0.60 | Singular dominance threshold |
| τ₂ | 0.25 | Normalized-entropy ceiling for singular emission |
| τ₃ | 0.65 | Entropy threshold → AMBIGUOUS |
| τ₄ | 0.40 | JSD threshold → BRANCH_CANDIDATE |

---

## Repository Contents

```
spl.py                              Core SPL implementation (SemanticUnit,
                                    tensor representation, EmissionEngine E0–E4,
                                    ClaimCandidateConverter)
spl_frontend.py                     Offline M1 fragmentation, M2 distributional
                                    type system, M3 typed relation projection
spl_gateway.py                      Protocol-callable interface layer
                                    (emit_claim_nodes, hash_claim, GatewayEvent)
WP2_Semantic_Projection_Layer.md    Full working paper (theory)

tests/
  test_entropy.py                   H_norm unit tests
  test_jsd.py                       JSD unit tests
  test_spl_rules.py                 Emission rules E0–E4 + end-to-end pipeline
  test_gateway.py                   Gateway boundary tests
  test_frontend.py                  Raw-text M1–M3 clean-room tests (DE/EN)

examples/
  simple_claim.txt                  "Paris is the capital of France." (E1)
  ambiguous_claim.txt               Modal+conjunctive hedging → E3 block
  multi_claim.txt                   3-unit sentence → mixed E1/E2 output

audit_log.json                      GatewayEvent log (auto-generated at runtime)
README.md                           This file
```

The reference implementation (`spl.py`) is also part of `hstre/Alexandria-Protokoll` as `alexandria_core/spl.py`.

---

## Status

This is a **working paper**. M1--M5 now have an executable offline reference
path. M1--M3 are deliberately a bounded rule baseline, not a claim of complete
natural-language understanding. Their purpose is to make the intermediate
representation, type uncertainty, relation-family filtering, illegal mass,
and sparse relational tensor executable and auditable before any learned
backend is introduced.

Open items:
- τ₂ calibration against gold-standard corpora
- Corpus evaluation and lexicon/grammar coverage beyond the DE/EN baseline
- Optional local learned backend behind the same explicit projection contract
- Evaluation against benchmark plan (Section 4 / WP2 Appendix I)

---

## Related

- [Alexandria Protocol](https://github.com/hstre/Alexandria-Protokoll) — the core protocol this paper extends
- SSRN submission pending

## License

Paper: CC BY 4.0  
Code: MIT

© 2026 Hanns-Steffen Rentschler
