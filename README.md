# Alexandria — Semantic Projection Layer

**Working Paper 2 · Formal Bridge between Natural Language and Epistemic Protocol**

> *The SPL transforms natural language segments into probabilistic relational tensors from which canonical claim candidates are derived through entropy-constrained projection rules.*

**Hanns-Steffen Rentschler · 2026**  
Part of the [Alexandria Protocol](https://github.com/hstre/Alexandria-Protokoll) ecosystem.

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
WP2_Semantic_Projection_Layer.md    Full paper (this working paper)
spl.py                              Reference implementation
README.md                           This file
```

The reference implementation (`spl.py`) is also part of `hstre/Alexandria-Protokoll` as `alexandria_core/spl.py`.

---

## Status

This is a **working paper** — the theory is stable, the implementation is complete, the NLP backend (embedding model integration) is pending.

Open items:
- τ₂ calibration against gold-standard corpora
- Production NLP backend (sentence-transformers / spaCy)
- Evaluation against benchmark plan (Section 4 / WP2 Appendix I)

---

## Related

- [Alexandria Protocol](https://github.com/hstre/Alexandria-Protokoll) — the core protocol this paper extends
- SSRN submission pending

## License

Paper: CC BY 4.0  
Code: MIT

© 2026 Hanns-Steffen Rentschler
