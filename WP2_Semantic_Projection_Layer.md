# Alexandria — Semantic Projection Layer

**Working Paper 2**  
*A Formal Bridge between Natural Language and Epistemic Protocol*

**Hanns-Steffen Rentschler** · Independent Researcher · 2026  
Alexandria Protocol v2.2 · Reference: `hstre/Alexandria-Protokoll`

---


## Abstract

*The Alexandria Protocol provides a formal framework for the versioned, auditable management of epistemic claims. However, the protocol presupposes that natural language has already been transformed into discrete, structured claims. This paper describes the Semantic Projection Layer (SPL), a formally defined pre-protocol stage that performs this transformation. The SPL fragments source text into minimal epistemic units, classifies their constituent entities by type, constrains candidate relations via a typed relation matrix, and computes probabilistic projections over the constrained relational space. Ambiguity is quantified using Shannon entropy, and builder divergence is measured by Jensen-Shannon divergence. The resulting claim candidates are structurally compatible with the Alexandria protocol without modifying its invariants. The SPL establishes a principled, language-independent bridge between natural language interpretation and epistemic versioning.*

## 1 Introduction

The Alexandria Protocol defines a rigorous framework for the creation, comparison, adjudication, and versioning of epistemic claims. Its core operations --- diff, adjudication, branching, and seal --- operate on discrete, structured claim objects. The protocol is, by design, agnostic about the origin of these claims.

In practice, however, claims originate from natural language: scientific papers, reports, databases, and datasets expressed in multiple languages with varying degrees of precision, hedging, and ambiguity. The transition from natural language to a structured claim is not trivial. Many expressions in scientific text do not map unambiguously onto a single relation type. Phrases such as "may suggest", "is associated with", or "legt nahe" occupy an intermediate space between multiple possible relational interpretations.

A direct mapping from text to claim --- without an intermediate interpretation stage --- produces three classes of problems:

-   Unstable claims: small variations in phrasing produce structurally different claims, generating artificial diffs.

-   Language-dependent artifacts: the same content expressed in different languages yields different claim structures.

-   Unrepresented ambiguity: epistemic hedging in the source text is discarded rather than preserved.

This paper introduces the Semantic Projection Layer (SPL) as a formally defined pre-protocol stage that addresses these problems. The SPL does not modify Alexandria's protocol invariants. It operates upstream of the canonical claim layer, transforming natural language segments into probabilistic relational structures from which claim candidates are derived.

The central claim of this paper is: *the SPL transforms natural language segments into probabilistic relational tensors from which canonical claim candidates are derived through entropy-constrained projection rules.*

## 2 Architecture Overview

The SPL is positioned as a distinct layer between raw source text and the canonical claim layer of the Alexandria Protocol. The complete epistemic pipeline is structured as follows:

> Layer 1 \| Source Text
>
> \| (papers, reports, datasets)
>
> \
>
> Layer 2 \| Semantic Projection Layer ← this paper
>
> \| (fragmentation, typing, projection, candidates)
>
> \
>
> Layer 3 \| Canonical Claim Layer
>
> \| (Alexandria claim objects)
>
> \
>
> Layer 4 \| Protocol Layer
>
> \| (Diff, Adjudication, Branch, Seal)
>
> \
>
> Layer 5 \| Knowledge Graph
>
> \
>
> Layer 6 \| Synapse Layer
>
> \| (graph comparison, structural alignment)

The SPL is pre-protocolar: its outputs are claim candidates, not sealed claims. The epistemic validity of all claims remains governed exclusively by the Alexandria protocol. The SPL controls only the quality and stability of the interpretation that precedes protocol operations.

The SPL defines three core operations, described in detail in the following sections:

-   Fragmentation: source text → minimal epistemic units (SemanticUnits)

-   Projection: SemanticUnit → probabilistic relational structure (SemanticProjection)

-   Emission: SemanticProjection → structured claim candidates (ClaimCandidates)

## 3 Semantic Units

### 3.1 Definition

A SemanticUnit is the smallest extractable text fragment that can carry a relational epistemic assertion. It is not a sentence, paragraph, or document section --- it is the minimal linguistic structure of the form:

> (entity) --- relation --- (entity \| property)

Formally, let U denote the space of all SemanticUnits. A fragment u ∈ U is a SemanticUnit if and only if it can be projected onto an interpretable relational structure with at least one subject candidate, one relation candidate, and one object candidate.

A single sentence may contain multiple SemanticUnits. This is the norm in scientific text, where complex sentences routinely bundle several relational assertions.

### 3.2 Fragmentation

Fragmentation is the process of decomposing source text into SemanticUnits. Boundary signals include relational verbs, modal constructions, evidential phrases, conjunctions, and scope markers. Examples of fragmentation boundary signals:

> Relational: causes, increases, decreases, enables, inhibits
>
> Modal: may, might, could, should
>
> Evidential: suggests, indicates, shows, supports
>
> Conjunctive: although, because, whereas, however
>
> Scope: in some sectors, under certain conditions

Example: the sentence "The results suggest that remote work may increase productivity, although the effect varies across sectors" yields three SemanticUnits:

> u1: results → suggest → \[claim: remote work increases productivity\]
>
> u2: remote work → may_increase → productivity
>
> u3: effect → varies_across → sectors

### 3.3 Formal Specification of the Projection Function π

This section provides the complete formal definition of the semantic projection function. The formulation is implementation-ready: all symbols are explicitly typed, all constraints are stated, and all decision rules are derived from Θ.

#### 3.3.1 Domain and Codomain

Let S be the set of all SemanticUnits. The projection function is:

> π : S → Π

where Π is the semantic projection space. Each element π(s) has the component structure:

> π(s) = (E_s, P_r, E_o, P_c, P_m, P_scope, U)

  --------------- ------------------------------------- -------------------------------------------------
  **Symbol**      **Type**                              **Semantics**

  **E_s**         **Set⟨Entity⟩**                       **Subject candidates extracted from s**

  **P_r**         **Distribution over ℛ**               **Relational distribution (marginal of R)**

  **E_o**         **Set⟨Entity⟩**                       **Object candidates extracted from s**

  **P_c**         **Distribution over ClaimCategory**   **Category: ONTIC, DYNAMIC, STATISTICAL ...**

  **P_m**         **Distribution over Modality**        **asserted / suggested / hypothesized ...**

  **P_scope**     **Distribution over Scope**           **Temporal, population, domain scope**

  **U**           **Uncertainty profile**               **Per-component entropy and confidence scores**
  --------------- ------------------------------------- -------------------------------------------------

#### 3.3.2 Relational Tensor

The core computational object is the relational tensor R, defined over subject candidates, permitted relations, and object candidates:

> R : E_s × ℛ × E_o → \[0, 1\]

where ℛ (calligraphic R) denotes the set of relations permitted by M for the type-pair (type(subject), type(object)). This notation distinguishes the relation space ℛ from the tensor R. Normalisation condition:

> Σ_i Σ_j Σ_k R(i, j, k) = 1

R is sparse: M constrains j to admissible relation families before projection, zeroing inadmissible entries. Marginal distributions:

> P_r(j) = Σ_i Σ_k R(i,j,k) \[relational marginal\]
>
> P_s(i) = Σ_j Σ_k R(i,j,k) \[subject marginal\]
>
> P_o(k) = Σ_i Σ_j R(i,j,k) \[object marginal\]

#### 3.3.3 Ambiguity Quantification

Ambiguity of the relational distribution is quantified by normalised Shannon entropy:

> H(P_r) = − Σ_j P_r(j) · log P_r(j)
>
> H_norm = H(P_r) / log(\|ℛ\|) ∈ \[0, 1\]

H_norm = 0: point mass on a single relation (fully determined). H_norm = 1: uniform over ℛ (maximally uninformative). Normalisation by log(\|ℛ\|) ensures comparability across SemanticUnits processed under different matrix versions with different \|ℛ\|.

#### 3.3.4 ClaimCandidate Generation

A ClaimCandidate is a triple:

> C = (s\*, r\*, o\*) ∈ E_s × ℛ × E_o

extracted from R by one of two operators, selected by the emission rules E1--E4 (Section 7):

> **E1 (singular):** C = argmaxᵣ R --- unique triple with highest R(i,j,k)
>
> **E2 (multiple):** C₁...C_k = top-k(R) --- k triples ranked by R(i,j,k)

Each ClaimCandidate inherits modality (from P_m), scope (from P_scope), and a back-reference to the originating SemanticUnit. The full π(s) is retained in the SemanticProjection record; no distributional information is discarded at emission.

#### 3.3.5 Builder Divergence

When two builders A, B independently project the same SemanticUnit, their relational distributions P_rᴬ and P_rᴭ may differ. Jensen--Shannon Divergence (JSD) quantifies this:

> JSD(P_rᴬ, P_rᴭ) = ½ KL(P_rᴬ ‖ M_mix) + ½ KL(P_rᴭ ‖ M_mix)
>
> M_mix = ½ (P_rᴬ + P_rᴭ)

JSD ∈ \[0,1\] when computed with base-2 logarithms (as in the reference implementation, Appendix I.6). It is symmetric, and is defined when one distribution assigns zero probability to a relation the other does not --- unlike KL-divergence. When JSD \> τ₄, emission rule E4 triggers a BRANCH_CANDIDATE rather than multi-emission (E2).

A dimension conflict arises when Builder A and Builder B assign different type classifications to the same subject or object, causing M to permit different relation families --- and thus different relational spaces ℛ\^A and ℛ\^B. JSD is undefined between distributions over different support sets.

Resolution: Global Simplex Embedding (Zero-Padding). Let ℛ_global be the union of all relations across all families in M. Each local distribution P_r\^A ∈ Δ(ℛ\^A) is embedded into the global simplex Δ\^(\|ℛ_global\|−1) by assigning probability 0 to all relations not permitted by M(t_s\^A, t_o\^A). Call this embedding P̅\_r\^A. The JSD is then computed over the global space:

> JSD(π_A(s), π_B(s)) = JSD(̅P_r\^A, ̅P_r\^B) where M = ½(̅P_r\^A + ̅P_r\^B)

If the permitted relation families are entirely disjoint (ℛ\^A ∩ ℛ\^B = ∅), the embedded vectors share no non-zero indices. In this case JSD(̅P_r\^A, ̅P_r\^B) = 1.0 (base-2 logarithm). This is not an edge case requiring special handling: a fundamental ontological disagreement between builders maximises JSD by construction, automatically triggering E4 and producing a BRANCH_CANDIDATE. No if-else branch is needed in the implementation.

*Implementation note: The global simplex embedding increases the dimension of the computation to \|ℛ_global\|, which is the full relation catalogue. For sparse distributions (typical in practice due to M constraints), this is efficient via sparse vector operations. The embedding must be consistent across both builders: both use the same ℛ_global ordering.*

#### 3.3.6 Formal Property and Extension Direction

Since π(s) encodes a probability distribution over relational triples, π maps into the probability simplex:

> π : S → Δ(E_s × ℛ × E_o)

where Δ(X) denotes the space of all probability distributions over X. This characterisation makes explicit that π does not produce a single relational assertion --- it produces a distribution over possible assertions. Epistemic commitment to a specific assertion arises only through the emission rules and the Alexandria protocol cycle.

*Π = Δ(E_s × ℛ × E_o) can be equipped with a metric derived from the JSD between projections, yielding a metric space structure on Π. This would enable semantic distance computation between SemanticUnits and inter-projection comparison for the Synapse layer. This direction is designated as an open problem for v2.3 (see also Section 12).*

### 3.4 Geometric Structure of the Semantic Projection Space Π

Section 3.3 established that π(s) ∈ Δ(E_s × ℛ × E_o). This section makes the geometric implications of that characterisation explicit. The structure of Π is not merely a formal curiosity: it is the foundation for semantic distance computation in the Synapse layer (v2.3) and for the geometric interpretation of ambiguity and builder divergence.

#### 3.4.1 The Probability Simplex

The space Δ(X) for a finite set X with \|X\| = n is the (n−1)-dimensional probability simplex:

> Δ\^(n-1) = { p ∈ ℝ\^n \| p_i ≥ 0, Σ_i p_i = 1 }

For the semantic projection space, X = E_s × ℛ × E_o. The dimension of Π is therefore:

> dim(Π) = \|E_s\| · \|ℛ\| × \|E_o\| − 1

Each vertex of the simplex corresponds to a degenerate distribution that assigns probability 1 to a single triple (s\*, r\*, o\*) --- a fully determined, unambiguous projection. Points in the interior correspond to distributions spread across multiple triples. Points on faces (not vertices) correspond to distributions confined to a subset of triples.

*For a SemanticUnit with 2 subject candidates, 4 permitted relations, and 2 object candidates, Π is a 15-dimensional simplex. Most practical projections inhabit a low-dimensional subspace due to sparsity from M.*

#### 3.4.2 Entropy as Geometric Position

The normalised Shannon entropy H_norm(π(s)) has a direct geometric interpretation: it measures the distance of the projection point from the nearest vertex of the simplex.

> H_norm = 0 ⇔ π(s) is a vertex (unambiguous, E1 applicable)
>
> H_norm = 1 ⇔ π(s) is the centroid (maximally ambiguous, E3 block)

The emission thresholds τ₂ and τ₃ partition the simplex into geometric regions:

> H_norm \< τ₂ → vertex neighbourhood (E1: single emission)
>
> τ₂ ≤ H_norm \< τ₃ → face region (E2: multiple emission)
>
> H_norm ≥ τ₃ → interior / centroid (E3: blocked)

This means the emission rules are geometric region selectors on Π --- not merely threshold comparisons on a scalar. The parameter set Θ defines the geometry of the decision boundary.

#### 3.4.3 Semantic Distance between Projections

Given two projections π(s₁) = P and π(s₂) = Q in Π, the Jensen--Shannon Divergence defines a pseudo-metric on Π:

> d_JSD(P, Q) = JSD(P, Q) ∈ \[0, 1\]

JSD is symmetric and bounded, but d_JSD is not a true metric: it does not satisfy the triangle inequality in general. However, its square root does:

> d(P, Q) = √JSD(P, Q)

√JSD is a metric on Δ(X) --- this follows from the fact that √JSD is equal to the Hellinger distance in the square-root parameterisation. The metric space (Π, √JSD) is therefore well-defined.

*This distinction matters for Synapse: if semantic distance computations require the triangle inequality (e.g. for clustering or nearest-neighbour retrieval), √JSD must be used, not raw JSD. The choice should be documented per use case.*

#### 3.4.4 Builder Divergence as Distance in Π

When two builders A and B project the same SemanticUnit s, their projections π_A(s) and π_B(s) are two points in Π. The E4 rule is a threshold on their distance:

> d_JSD(π_A(s), π_B(s)) \> τ₄ → BRANCH_CANDIDATE

This gives branching a geometric justification: a branch is created when two builders locate the same SemanticUnit in regions of Π that are too distant to be reconciled by multi-emission (E2). The branch boundary is a hypersphere of radius τ₄ around each projection point in Π.

#### 3.4.5 Semantic Manifold and Drift Detection

When a corpus of SemanticUnits s₁, ..., s_n is projected, the resulting set {π(s₁), ..., π(s_n)} forms a point cloud in Π. Under regularity conditions on the corpus, this point cloud approximates a smooth submanifold M ⊂ Π --- a semantic manifold of the corpus.

This has two concrete consequences for downstream applications:

• Similar claims cluster: SemanticUnits that assert similar propositions produce projections that are geometrically close in Π. This enables similarity search over SemanticProjections without committing to discrete claim structures.

• Semantic drift is detectable: If the same term or concept produces projections that migrate in Π over time (e.g. across corpus epochs), this is observable as displacement of the point cloud --- without requiring comparison of sealed claims.

*The manifold hypothesis for Π is an empirical claim that must be validated on real corpora. It is stated here as a research direction, not as an established property of the SPL.*

#### 3.4.6 Relation to the Synapse Layer

The geometric structure of Π established in this section is a prerequisite for the Synapse layer (Paper 3). Synapse compares not only sealed claims (discrete, versioned) but also the underlying projections that generated them. The metric (Π, √JSD) enables:

• Comparison of two projections of the same SemanticUnit across builders or time.

• Definition of a semantic distance between two SemanticUnits based on their projections.

• Graph-level distance between two knowledge corpora, as a function of the distances between their projection point clouds (Hausdorff or Wasserstein distance over Π).

*The formal definition of inter-graph semantic distance is designated as the central open problem of Paper 3 (Synapse). The present section establishes the geometric foundation that makes such a definition possible.*

### 3.5 Semantic Distance between Claim Graphs: Transition to Synapse

This section defines the formal distance measures between individual claims and between claim graphs. These definitions constitute the mathematical interface between Paper 2 (SPL) and Paper 3 (Synapse). They are stated here because they depend directly on the geometric structure of Π established in Section 3.4; they are not yet the full Synapse specification.

#### 3.5.1 Projection-Extended Claim Representation

In the standard Alexandria protocol, a sealed claim is a triple C = (s, r, o) with associated metadata. For inter-graph comparison, each claim is extended with its originating semantic projection:

> C⁺ = (s, r, o, P) where P = π(s₀) ∈ Π

and s₀ is the SemanticUnit from which C was extracted. P is the full relational distribution over E_s × ℛ × E_o --- not only the marginal P_r. The extended claim C⁺ retains the provenance of the discrete claim C: how certain the builder was, what alternatives were present, and which matrix version was active. This information is available without any additional computation since SemanticProjection records are retained (Section 3.3.4).

#### 3.5.2 Distance between Two Claims

Let C₁⁺ = (s₁, r₁, o₁, P₁) and C₂⁺ = (s₂, r₂, o₂, P₂) be two extended claims. Their distance is defined as a weighted combination of three components:

> d_claim(C₁⁺, C₂⁺) = α · d_entity(s₁, s₂, o₁, o₂) + β · d_rel(r₁, r₂) + γ · d_sem(P₁, P₂)

Subject to the normalisation constraint:

> α + β + γ = 1, α, β, γ ≥ 0

The three components are:

**d_entity:** Entity distance. Measures similarity between subject and object entities across the two claims. Implementation options include embedding cosine distance, ontology path distance, or string similarity --- depending on the knowledge domain. d_entity is symmetric and must be normalised to \[0,1\].

**d_rel:** Relation distance. Measures dissimilarity between the two relation types r₁, r₂. This can be derived from the relation family hierarchy (e.g. increases and causes are both DYNAMIC, distance small; increases and correlates_with cross family boundaries, distance larger) or from a trained relation embedding.

**d_sem:** Semantic distribution distance. Measures the distance between the underlying projections in Π. The metric is √JSD (Section 3.4.3): d_sem(P₁, P₂) = √JSD(P₁, P₂) ∈ \[0,1\].

*The weights α, β, γ are domain-specific calibration parameters analogous to Θ. No universal values are proposed here. Their calibration requires annotated claim pairs with known human similarity judgements --- an empirical task outside the scope of Paper 2.*

#### 3.5.3 Contradiction vs. Interpretation Difference

The three-component distance enables a distinction that standard knowledge graphs cannot make:

> • Structural contradiction: d_entity small (same entities), d_rel large (opposing relations, e.g. increases vs. decreases). The claims refer to the same subject-object pair but assert incompatible relations. Detectable from d_rel alone.
>
> • Interpretive divergence: d_entity small, d_rel moderate (adjacent relations, e.g. increases vs. correlates_with), d_sem large (P₁ and P₂ are far apart in Π). The claims assert similar relations but the builders had very different distributional confidence. This is a disagreement in interpretation, not in the asserted fact.
>
> • Thematic distance: d_entity large. The claims address different subject-object pairs entirely. Graph-level comparison is required.

This three-way decomposition is the principal advantage of projection-extended claims over discrete knowledge graph triples. A system using only C = (s, r, o) cannot distinguish case 1 from case 2.

#### 3.5.4 Distance between Two Claim Graphs

A claim graph G is a set of extended claims:

> G = { Cⁱ⁺ \| i = 1, ..., \|G\| }

Given two graphs G₁ and G₂, the graph distance D(G₁, G₂) is defined via optimal matching. Let M ⊆ G₁ × G₂ be a matching (each claim matched to at most one claim in the other graph). The graph distance is:

> D(G₁, G₂) = min_M ¹⁄\|M\| Σ\_{(C_i, D_j) ∈ M} d_claim(C_i⁺, D_j⁺)

where the minimum is taken over all valid matchings M.

*Computational complexity: optimal bipartite matching over \|G₁\| × \|G₂\| pairs is O(n³) via the Hungarian algorithm. For graphs with \|G\| \> \~500 claims, this becomes computationally intractable in practice. Two approximation strategies are available: (1) Wasserstein-1 distance with an efficient solver (e.g. Sinkhorn iterations), treating the claim sets as discrete measures in Π weighted by candidate_score; or (2) random subsampling of the matching. The choice of approximation is an implementation decision and is designated as an open problem for Paper 3.*

#### 3.5.5 Wasserstein Distance as Graph Distance (Alternative Formulation)

The Wasserstein-1 distance (Earth Mover's Distance) over the metric space (Π, √JSD) provides a tractable alternative to optimal matching for large graphs. Given two empirical measures:

> μ₁ = ¹⁄\|G₁\| Σ_i δ\_{π(s_i)} (uniform over projections in G₁)
>
> μ₂ = ¹⁄\|G₂\| Σ_j δ\_{π(s_j)} (uniform over projections in G₂)

the Wasserstein-1 distance is:

> W₁(μ₁, μ₂) = inf\_{γ ∈ Γ(μ₁,μ₂)} ∫ √JSD(x,y) · dγ(x,y)

where Γ(μ₁, μ₂) is the set of all couplings of μ₁ and μ₂. This formulation treats a claim graph as a probability measure over the projection space Π, and the distance between graphs as the minimum transport cost to move one measure onto the other.

*The Wasserstein formulation has two advantages: it is approximable in polynomial time via the Sinkhorn algorithm, and it naturally handles graphs of different sizes without requiring an explicit matching. Its disadvantage is that it uses only the projection component d_sem and ignores d_entity and d_rel --- it is a purely geometric distance in Π. Whether this is sufficient for Synapse's comparison tasks is an open empirical question.*

*Scope restriction: The Wasserstein approximation over Π is semantically valid only when G₁ and G₂ are bounded to the same entity subsets --- for example, two branch nodes derived from the same source corpus. It cannot serve as a global distance metric between unconstrained distinct corpora, since the point clouds μ₁ and μ₂ would then mix different entity domains, making the transport cost geometrically meaningful but epistemically uninterpretable.*

#### 3.5.6 Open Problems Deferred to Paper 3 (Synapse)

The following questions are defined here but not resolved:

> • Calibration of α, β, γ: What weighting of entity, relation, and semantic components produces claim distances that correspond to human similarity judgements? This requires annotated evaluation data.
>
> • Approximation selection: Under what graph size and precision requirements should bipartite matching, Wasserstein distance, or subsampling be used?
>
> • Epistemic curvature: Is there a natural Riemannian structure on Π (e.g. Fisher--Rao metric) that provides a more informative geometry than the flat simplex? If so, curvature in Π could serve as a measure of internal coherence of a knowledge corpus. This is a research direction, not a defined property of the current system.
>
> • Cross-branch graph distance: How should D(G₁, G₂) be computed when G₁ and G₂ were validated under different matrix versions (structural_matrix_mismatch, Appendix K)? The distance is not directly defined in this case.

*These four problems constitute the formal research agenda for Paper 3.*

### 3.6 Information Geometry of Π and Epistemic Curvature

This section establishes two related results. First, the Semantic Projection Space Π carries a natural information-geometric structure derived from the Fisher--Rao metric --- a standard result from information geometry applied here to claim structures. Second, this structure enables an operational definition of epistemic curvature K(G) as a measure of semantic divergence within a claim graph, with a direct connection to the Alexandria protocol's sealing condition.

#### 3.6.1 Π as a Statistical Manifold

The probability simplex Δ\^(n−1) is a well-studied object in information geometry (Amari, 1985; Amari & Nagaoka, 2000). The interior of Δ\^(n−1) --- the set of strictly positive distributions --- is a smooth (n−1)-dimensional manifold, denoted S\^(n−1). This manifold admits a natural Riemannian metric: the Fisher--Rao metric.

For a distribution P = (p₁, ..., p_n) ∈ S\^(n−1), the Fisher information matrix is:

> g\_{ij}(P) = Σ_k ¹⁄p_k · ∂p_k⁄∂θ_i · ∂p_k⁄∂θ_j

where θ = (θ₁, ..., θ\_{n−1}) is a local coordinate system on S\^(n−1). The geodesic distance under this metric is the Fisher--Rao distance d_FR(P, Q). For discrete distributions, d_FR has a closed form via the Bhattacharyya angle:

> d_FR(P, Q) = 2 · arccos( Σ_k √(p_k · q_k) )

This metric provides a principled, coordinate-free notion of distance on Π that is invariant under reparametrisation of the underlying relational space --- a property that JSD does not have. The connection between the two is: √JSD ≤ d_FR/√2 (Endres & Schindelin, 2003), so the two metrics are topologically equivalent on compact subsets of Π.

*This paper does not propose Fisher--Rao as the primary distance metric for practical computation. JSD and √JSD remain the operational metrics for Sections 3.5--3.7. Fisher--Rao is introduced here to establish that Π carries a well-defined Riemannian structure, which justifies the use of curvature terminology in Section 3.4.3.*

#### 3.6.2 Claim Extraction as Projection onto Discrete Structure

The emission step π(s) → C = (s\*, r\*, o\*) can be interpreted geometrically as a projection from the continuous manifold S\^(n−1) onto its boundary and vertices. Specifically:

> • E1 (argmax extraction): maps P to the nearest vertex of Δ\^(n−1) in the L∞ sense. This is a projection onto the extreme points of the simplex.
>
> • E2 (top-k extraction): maps P to a face of dimension k−1, the face spanned by the k relations with highest probability mass.
>
> • E3 (block): retains P in the interior; no projection is performed.

This framing makes explicit that claim extraction is a lossy operation: the projection discards the distributional information in P and retains only the discrete triple. The extended claim representation C⁺ = (s, r, o, P) from Section 3.3.1 is therefore a partial recovery of this lost information.

#### 3.6.3 Epistemic Curvature K(G)

Given a claim graph G with projection-extended claims Cⁱ⁺ = (s_i, r_i, o_i, P_i), the local epistemic curvature at claim C_i is defined as the average JSD between P_i and its neighbourhood N_i in the graph:

> K_i = ¹⁄\|N_i\| Σ\_{j ∈ N_i} JSD(P_i, P_j)

Dimension conflict also affects K(G): if C_i is a STATISTICAL claim and its graph-neighbour C_j is a NORMATIVE claim, their projections P_i and P_j inhabit different relational subspaces. JSD between them is either undefined or semantically meaningless --- it would measure ontological diversity, not epistemic tension.

Resolution: Homologous Neighborhoods. We restrict N_i to the homologous neighbourhood N_i\*:

> N_i\* = { j ∈ N_i \| type_sig(C_j) = type_sig(C_i) and (C_j, C_i) ∈ E }

where type_sig(C) = M(type(subject), type(object)) --- the matrix cell that permitted C's relation. Two claims are homologous if and only if they were validated against the same matrix cell. The revised local curvature is:

> K_i = ¹⁄\|N_i\*\| Σ\_{j ∈ N_i\*} JSD(P_i, P_j) if N_i\* ≠ ∅
>
> K_i = 0 if N_i\* = ∅

The global graph curvature K(G) is recomputed as the mean over all K_i. Under this definition, K(G) measures genuine epistemic tension --- competing interpretations of the same relational context --- rather than the mere fact that a knowledge graph contains claims of different ontological categories. A graph that connects chemical process claims with legal norm claims has K(G) = 0 between those categories; K(G) is only elevated when same-type claims diverge in their projections.

*The ESF from Section 3.4.5 inherits this fix: the K(G) term in S(G) now uses homologous neighbourhoods. The four-factor formula is unchanged; only the computation of K_i is restricted to N_i\*.*

> K(G) = ¹⁄\|E\| Σ\_{(i,j) ∈ E} JSD(P_i, P_j)

Interpretation: K(G) ∈ \[0, 1\]. Low K(G) indicates that connected claims have similar projection distributions --- i.e. the claims arose from linguistically similar, mutually consistent assertions. High K(G) indicates that the graph connects claims whose projections are far apart in Π, which is a signature of semantic tension or genuine contradiction.

Concrete example: Early pandemic literature on mask efficacy contained claims ranging from „ineffective" to „strongly effective". The corresponding projections P_i would be distributed across widely separated regions of Π. K(G) would be high. As the scientific consensus converged, the point cloud contracted, and K(G) decreased.

*Relation to Ollivier--Ricci curvature: Ollivier (2009) defines a curvature on graphs using the Wasserstein distance between local neighbourhoods of connected nodes. The epistemic curvature K_i defined here is structurally analogous: it computes a distributional distance (JSD rather than Wasserstein-1) between the projections of connected claims. The difference is that Ollivier--Ricci curvature operates on the graph topology itself, while K_i operates on the semantic content of the nodes. Both quantities can in principle be computed simultaneously, and their comparison is an open question for Paper 3.*

#### 3.6.4 Curvature in the Fisher--Rao Manifold

The Riemannian curvature of the probability simplex S\^(n−1) under the Fisher--Rao metric is constant and positive: S\^(n−1) is isometric to an (n−1)-dimensional sphere of radius 2 (under the square-root parameterisation φ_i = √p_i). This is a standard result.

The epistemic curvature K(G) defined in 3.6.3 is not the same as the Riemannian curvature of S\^(n−1). The latter is a fixed property of the space. K(G) is an operational measure of the spread of a specific point set (the claim projections) within that space. The two are related: K(G) can be interpreted as a discrete approximation to the average length of geodesic arcs between neighbouring points on the semantic manifold of the corpus.

*This distinction is important for reviewers from differential geometry: we are not claiming that the claim graph induces new curvature on the manifold. We are using the manifold structure to give a geometric interpretation to divergence between claims.*

#### 3.6.5 Epistemic Stability Function (Extended Form)

The connection to the Alexandria protocol arises through the sealing condition. Section 3.4.3 introduced the single-factor form S(G) = 1 − K(G). Here we define the full Epistemic Stability Function (ESF), which combines four independent factors. Sealing is a function of all four.

**Factor 1 --- Semantic Consistency K(G).** Epistemic curvature as defined in 3.6.3. Measures the average JSD between neighbouring claims in the graph. K(G) ∈ \[0,1\] by construction (JSD is bounded).

> K(G) = ¹⁄\|E\| Σ\_{(i,j)∈E} JSD(P_i, P_j)

**Factor 2 --- Projection Entropy H(G).** Average normalised Shannon entropy across all claims. A claim with high H_i is semantically ambiguous regardless of its neighbours. H(G) ∈ \[0,1\] since H_i = H_norm ∈ \[0,1\].

> H(G) = ¹⁄\|C\| Σ_i H_i where H_i = H_norm(π(s_i))

**Factor 3 --- Builder Convergence D(G).** Average JSD between builder A and builder B projections across all claims. D(G) = 0 if all builders agree; D(G) = 1 if maximally divergent. Defined only in multi-builder deployments; D(G) = 0 by convention in single-builder mode.

> D(G) = ¹⁄\|C\| Σ_i JSD(P_i\^A, P_i\^B)

**Factor 4 --- Structural Completeness U(G).** Fraction of unresolved protocol operations (open Diffs, pending Adjudications, unresolved BranchNodes) relative to total claims. U(G) = 0 when all operations are resolved; U(G) = 1 when no claims have been processed. U(G) ∈ \[0,1\] by construction.

> U(G) = open_diffs + pending_adjudications + unresolved_branches / \|C\

*U(G) can exceed 1 if multiple unresolved operations exist per claim. In practice, U(G) should be capped at 1 for the ESF computation.*

The Epistemic Stability Function is defined as:

> S(G) = 1 − (α K(G) + β H(G) + γ D(G) + δ U(G))

Subject to the normalisation constraints:

> α + β + γ + δ = 1, α, β, γ, δ ≥ 0

Since each factor is in \[0,1\] and the weights sum to 1, the weighted sum is also in \[0,1\], and therefore S(G) ∈ \[0,1\]. S(G) = 1 indicates a maximally stable graph: zero curvature, zero entropy, full builder agreement, no open operations. S(G) = 0 indicates maximum instability on all four dimensions simultaneously.

The seal condition for a claim graph G is:

> S(G) ≥ τ_seal

and the local seal condition for an individual claim C_i (used in incremental sealing) is:

> S(C_i) = 1 − (α K_i + β H_i + γ D_i + δ U_i) ≥ τ_seal

where K_i, H_i, D_i are the local versions of each factor for claim C_i, and U_i = 1 if C_i has any unresolved protocol operations, 0 otherwise.

*Θ is now extended to Θ = {τ₁, τ₂, τ₃, τ₄, τ_seal, α, β, γ, δ}. Conservative defaults: τ_seal = 0.75; α = 0.3 (curvature), β = 0.2 (entropy), γ = 0.2 (builder divergence), δ = 0.3 (structural completeness). These defaults are not empirically calibrated and should be treated as a starting configuration pending corpus-based validation.*

The ESF gives Seal a semantic grounding: a claim cannot be sealed simply because no one has objected to it. It can only be sealed when the graph is geometrically consistent (low K), interpretively clear (low H), builder-convergent (low D), and protocol-complete (low U). This is a significantly stronger condition than the existing adjudication-based sealing criterion, and constitutes the primary formal contribution of this section to the Alexandria protocol.

*Integration with Synapse: S(G) is a scalar summary of the epistemic state of a knowledge graph at a given moment. Synapse can track S(G_A) and S(G_B) over time, compare their stability trajectories, and flag when two graphs that were previously similar diverge in stability --- a potential signature of paradigm conflict.*

**Epistemic Stringency as Governed Parameter.** The weight vector Θ_ESF = {α, β, γ, δ, τ_seal} defines the Epistemic Stringency of the system: how much curvature, entropy, builder divergence, and structural incompleteness is tolerated before a graph is considered stable enough to seal. Different scientific domains tolerate different thresholds --- a clinical trial corpus may require near-zero builder divergence (γ high), while an exploratory philosophy corpus may tolerate high entropy (β low).

Θ_ESF cannot therefore be globally hardcoded. Instead, it is treated as a governed parameter: Θ_ESF is cryptographically bound to the BranchNode alongside the structural_context (Appendix K.1). The genesis BranchNode initialises with the conservative defaults specified above. Any adjustment to the stringency profile requires a StringencyPatch to be submitted to the protocol, audited via the three-level audit (J.2), and approved by builder consensus.

If consensus on a StringencyPatch is not reached (JSD \> τ₄), the stringency change itself becomes a BRANCH_CANDIDATE: two branches of the graph proceed under different epistemic stringency profiles. This is not a failure mode --- it is the correct representation of genuine disagreement about what "stable" means in a given domain.

*This converts a documentation weakness ("parameters are not empirically calibrated") into an explicit design feature ("the protocol is built so that calibration is a governed, versioned, community process"). The formal status of default values changes from provisional to axiomatic: the genesis stringency profile is an axiom of the genesis BranchNode, exactly as T_v1.0 and M_v1.0 are axioms of the genesis vocabulary and matrix.*

#### 3.6.6 Contribution Statement for Reviewers

The contribution of this section is not a new geometry. The Fisher--Rao metric and information geometry are established (Amari, 1985). The contribution is the application of this structure to epistemic claim analysis:

> • The identification of Π as a statistical manifold under the Fisher--Rao metric, which justifies curvature-based analysis of claim graphs.
>
> • The operational definition of K(G) as a scalar measure of semantic divergence within a claim graph, computed via JSD over projection neighbourhoods.
>
> • The Epistemic Stability Function S(G) = 1 − K(G) and the derived local seal condition, which connects the geometric structure of Π to the Alexandria protocol's sealing semantics.
>
> • The explicit connection to Ollivier--Ricci curvature as an open comparison problem for Paper 3.

The proposed abstract sentence: The Semantic Projection Layer embeds linguistic interpretations into a probability simplex over relational triples. This representation induces an information-geometric structure that allows epistemic claim graphs to be analysed using information-theoretic distance and curvature measures, with direct operational consequences for the Alexandria protocol's sealing condition.

## 4 Type System

### 4.1 Type Classes

Every entity or property extracted from a SemanticUnit is assigned a type class. The type system defines a finite set T of object classes:

  ---------------------- --------------------------------------------------------
  Entity                 A named thing, person, organization, or substance

  Property               A measurable or observable attribute

  Process                An ongoing activity, mechanism, or procedure

  Event                  A discrete occurrence with temporal extent

  Population             A group of entities sharing a defining characteristic

  Variable               A quantity that takes different values across contexts

  Model                  A formal or computational representation of a system

  Evidence               An empirical observation, study, or dataset

  Claim                  A propositional assertion about the world

  Norm                   A rule, regulation, guideline, or obligation
  ---------------------- --------------------------------------------------------

Type assignment is context-dependent. The same token may receive different types in different contexts. The classification function is therefore defined over (token, context_window) pairs, not over tokens in isolation.

The type system defines a finite but protocol-extensible set T of object classes. The genesis version (T_v1.0) defines the ten foundational classes listed above. Like the relational matrix M, T is not permanently hardcoded. New type classes can be proposed, audited, and sealed into the system vocabulary via TypePatch operations, ensuring the ontology can evolve alongside scientific paradigms. The governance mechanism is defined in Appendix J.7.

*T_v1.0 is an axiom, set outside the governance cycle for the same reason as M_v1.0 (see J.5.1). All subsequent extensions are governed. The append-only constraint (J.7) ensures that no historical claims lose their type signature when T evolves.*

### 4.2 Type Classification

Type classification is performed via a distributional model over the type space T. For a token t in context window w, the classifier produces a probability distribution:

> P_T(type \| t, w) → distribution over T

The assigned type is the argmax of this distribution. If the distribution entropy exceeds a threshold, the token is flagged as AMBIGUOUS_TYPE and the fragmentation may be revised or the unit flagged for dual-builder treatment.

## 5 Relation System

### 5.1 Relation Families

Relations are organized in a two-level hierarchy: families and concrete relations. This hierarchy serves two purposes: it reduces the projection search space, and it enforces the separation of ontologically distinct relation categories.

Six relation families are defined:

> ONTIC has_property, part_of, located_in, participates_in, instance_of
>
> DYNAMIC affects, increases, decreases, stabilizes, causes, enables,
>
> inhibits, triggers
>
> STATISTICAL correlates_with, covaries_with, associates_with, differs_from
>
> EPISTEMIC supports, contradicts, suggests, refines, extends, qualifies
>
> MODEL predicts, explains, estimates, simulates, approximates
>
> NORMATIVE requires, forbids, recommends, permits, prioritizes

The separation of STATISTICAL from DYNAMIC is a deliberate design decision. Statistical association does not imply causal mechanism. Conflating correlates_with with increases or causes is one of the most common errors in automated knowledge graph construction. The SPL enforces this distinction structurally.

### 5.2 Typed Relation Matrix

The relation space is not treated as an unconstrained set. A typed relation matrix M maps pairs of type classes to permitted relation families:

> M : T × T → 2\^F

where F is the set of relation families. A representative subset of the matrix:

  ------------------ ----------------- ------------------------------------
  **Subject Type**   **Object Type**   **Permitted Relations**

  Process            Property          DYNAMIC, STATISTICAL

  Evidence           Claim             EPISTEMIC

  Model              Event             MODEL

  Norm               Process           NORMATIVE

  Entity             Property          ONTIC, DYNAMIC

  Process            Process           DYNAMIC

  Evidence           Evidence          EPISTEMIC, STATISTICAL

  Variable           Variable          STATISTICAL, DYNAMIC
  ------------------ ----------------- ------------------------------------

The matrix acts as a structural filter: the projection layer computes relation probabilities only within the family set opened by the type combination of subject and object. Relations outside this set are structurally inadmissible for that unit.

### 5.3 Two-Stage Projection

Because the relation space is hierarchically organized, the projection operates in two stages. This factorization reduces error and makes the projection interpretable:

> Stage 1 (family): P(f \| u) over M(t_s, t_o)

To prevent the SPL from forcing linguistically explicit but ontologically invalid assertions into permitted categories, Stage 1 is computed over the unconstrained set of all relation families ℱ, not only M(t_s, t_o):

> Stage 1 (global family): P(f \| u) over all f ∈ ℱ
>
> Stage 2 (local relation): P(r \| f, u) over relations within permitted f ∈ M(t_s, t_o) \[unchanged\]

This separation allows the SPL to detect when the primary linguistic signal of the source text falls outside the structurally permitted relational space. The unpermitted mass P_illegal = Σ\_{f ∉ M(t_s, t_o)} P(f \| u) is then used by Rule E0 (see Section 7.2) as a pre-flight rejection criterion. Without Stage 1 over ℱ, Rule E0 cannot be computed.

> Stage 2 (relation): P(r \| f, u) over relations in f
>
> Full projection: P(r \| u) = P(f \| u) · P(r \| f, u)

The first stage selects the relation family; the second stage resolves the specific relation within that family. A model that produces SUGGESTS before committing to supports versus contradicts is making two epistemically distinct decisions.

## 6 Semantic Projection

### 6.1 Projection Function

The projection function π maps each SemanticUnit to a SemanticProjection:

> π : U → Π
>
> Π = (subject_candidates,
>
> relation_distribution,
>
> object_candidates,
>
> category_distribution,
>
> modality_distribution,
>
> scope_candidates,
>
> uncertainty_profile)

The relation_distribution is a probability distribution over the relation set opened by M(t_s, t_o). The modality_distribution is a separate distribution over the modality space:

> Modality space M_modal:
>
> asserted, supported, suggested, hypothesized,
>
> possible, disputed, negated

### 6.2 Tensor Representation

The full projection can be expressed as a relational tensor R indexed by subject candidate, relation, and object candidate:

> R(i, j, k) = P(subject=i, relation=j, object=k \| u)
>
> Example:
>
> R(RemoteWork, increases, Productivity) = 0.31
>
> R(RemoteWork, correlates_with, Productivity) = 0.42
>
> R(RemoteWork, affects, Productivity) = 0.18

Extended with modality and time dimensions, this becomes:

> R(t_s, t_o, r, m, τ)
>
> t_s = subject type
>
> t_o = object type
>
> r = relation
>
> m = modality
>
> τ = temporal scope

### 6.3 Orthogonal Temporal Dimension (τ)

Temporal ordering relations (precedes, during, after, overlaps) are fundamentally distinct from epistemic, dynamic, or any other relation family. The distinction is not merely taxonomic --- it is mathematical. Causality inherently implies temporal precedence: a cause precedes its effect. If temporal relations are included as elements of ℛ alongside DYNAMIC relations such as causes, the SPL is forced to distribute probability mass across both. A text that unambiguously asserts „X caused Y three hours later" would produce a distribution split between causes ≈ 0.45 and precedes ≈ 0.45, yielding H_norm ≈ 0.69 and triggering Rule E3 (AMBIGUOUS) --- a false negative on a perfectly clear assertion.

Temporal relations are therefore excluded from the primary relation families ℱ. Instead, they constitute an orthogonal dimension of the semantic projection tensor. The extended tensor is:

> R : E_s × ℛ × E_o × ᴺ\_{Allen} → \[0, 1\]

where ᴺ\_{Allen} is a discrete set of temporal interval relations drawn from Allen's Interval Algebra (Allen, 1983): {before, meets, overlaps, starts, during, finishes, equals, unspecified}. The temporal dimension is orthogonal to the relational dimension: the SPL computes P(r \| u) and P(τ \| u) independently.

The normalisation conditions are separate:

> Σ_r P(r \| u) = 1 \[over permitted relations in ℛ\]
>
> Σ_τ P(τ \| u) = 1 \[over Allen interval relations\]

This separation preserves the entropy calculation: H_norm is computed only over P(r \| u). The temporal distribution P(τ \| u) does not contribute to H_norm and therefore cannot trigger artificial ambiguity blocks. A claim that asserts causes with P = 0.72 and before with P = 0.91 receives status READY_FOR_CLAIM (E1) on the relational dimension regardless of the temporal distribution. The temporal_ordering field is a first-class component of the ClaimCandidate: it is included in the Canonical Identity Key (Section 10.2), so two claims with identical subject, relation, object, and scope but different temporal_ordering are distinct propositions in the protocol.

*Allen's Interval Algebra is used here at the level of granularity appropriate for natural language: the full 13-relation algebra is reduced to 8 canonical relations plus unspecified. The unspecified value is the default when no temporal signal is present in the source text, not an indication of ambiguity. Temporal ambiguity (e.g., uncertainty whether X precedes or overlaps Y) is represented as a distribution over ᴺ\_{Allen} with high entropy, independently of the relational projection.*

## 7 Ambiguity Quantification and Emission Rules

### 7.1 Shannon Entropy as Ambiguity Measure

Ambiguity in the semantic projection is quantified using Shannon entropy over the relation distribution. Shannon entropy is chosen because it captures distributional uncertainty across the full projection space, penalizing both sharply peaked and flat distributions in a theoretically principled and well-understood way. Alternative measures such as Gini impurity are optimized for classification contexts rather than probability distributions over relational spaces; energy-based scores require a generative model not defined here.

> H(P_r) = - Σ_i p_i · log(p_i)
>
> Normalized: H_norm = H / log(\|R\|)
>
> \|R\| = number of candidate relations
>
> H_norm ∈ \[0, 1\]
>
> H_norm = 0 → fully determined projection
>
> H_norm = 1 → maximally ambiguous projection

### 7.2 Emission Rules

Before the four emission rules E1--E4 are applied, a pre-flight structural check is performed. This check depends on the global Stage 1 projection P(f \| u) over all families ℱ (Section 5.3):

> P_illegal = Σ\_{f ∉ M(t_s, t_o)} P(f \| u)
>
> Rule E0 --- Structural Rejection
>
> IF P_illegal \> τ₀
>
> THEN status = STRUCTURAL_VIOLATION, no emission

τ₀ ∈ Θ is the structural rejection threshold. Recommended initial value: τ₀ ≈ 0.50. If the model judges it more likely than not that the text asserts a relation from a forbidden family, the SemanticUnit is rejected before any E1--E4 processing.

Rationale: Rule E0 is an ontological shield. Consider the assertion „A statute (Norm) oxidises (DYNAMIC) a molecule (Entity)". M does not permit DYNAMIC relations between Norm and Entity. Without E0, the SPL would either (a) redistribute the probability mass over permitted but semantically incorrect relations (e.g. requires), or (b) generate high entropy and block via E3 (AMBIGUOUS). Neither correctly represents the source: AMBIGUOUS implies a recoverable interpretation problem; this is a categorical error in the text. STRUCTURAL_VIOLATION is the correct epistemic status.

*STRUCTURAL_VIOLATION is a new status added to the protocol alongside PROJECTED, AMBIGUOUS, READY_FOR_CLAIM, and BRANCH_CANDIDATE. Persistent STRUCTURAL_VIOLATIONs for a given type-pair are a data-driven signal that M may be too restrictive and should be updated via a RULE_3_EXTEND patch through the governance cycle (Appendix J). This makes the system organically self-correcting: structural errors in the corpus drive matrix evolution.*

Θ is extended: Θ = {τ₀, τ₁, τ₂, τ₃, τ₄, τ_seal, α, β, γ, δ}. τ₀ ∈ (0.5, 1). The emission rules E1--E4 apply only when E0 has passed (P_illegal ≤ τ₀).

Four emission rules govern the transition from SemanticProjection to ClaimCandidates:

> Rule E1 --- Singular emission
>
> IF max(P_r) \> τ1 AND H_norm \< τ2
>
> THEN emit single ClaimCandidate (argmax)
>
> Rule E2 --- Multiple emission
>
> IF max(P_r) ≤ τ1 AND H_norm \< τ3
>
> THEN emit top-k ClaimCandidates with scores
>
> Rule E3 --- Ambiguity block
>
> IF H_norm ≥ τ3
>
> THEN status = AMBIGUOUS, no emission
>
> Rule E4 --- Builder divergence branch
>
> IF JSD(P_r\^A, P_r\^B) \> τ4
>
> THEN status = BRANCH_CANDIDATE

### 7.3 Divergence Between Builders

When two builders produce projections for the same SemanticUnit, their divergence is measured using Jensen-Shannon divergence (JSD). JSD is preferred over KL-divergence because it is symmetric, bounded on \[0, 1\], and well-defined when one distribution assigns zero probability to an outcome.

> JSD(P, Q) = (1/2) · KL(P \|\| M) + (1/2) · KL(Q \|\| M)
>
> M = (P + Q) / 2
>
> JSD = 0 → identical projections
>
> JSD ≈ 0.2 → minor divergence
>
> JSD ≈ 0.5 → substantial divergence → BRANCH_CANDIDATE

### 7.4 Configuration Parameters

The emission thresholds constitute the configuration parameter set of the SPL:

> Θ = {τ1, τ2, τ3, τ4}
>
> τ1 ∈ (1/\|R\|, 1) dominance threshold
>
> τ2 ∈ (0, 0.5) low-entropy ceiling for singular emission
>
> τ3 ∈ (0.5, 1) high-entropy floor for ambiguity block
>
> τ4 ∈ (0, 1) builder divergence threshold for branch
>
> Recommended initial values:
>
> τ1 ≈ 0.55 -- 0.65
>
> τ2 ≈ 0.20 -- 0.30
>
> τ3 ≈ 0.60 -- 0.70
>
> τ4 ≈ 0.35 -- 0.45

These parameters are not part of the theory. They are calibration parameters of the SPL implementation. Their bounds are derived from the structure of the probability space; their specific values are calibrated empirically against annotated corpora or optimized for inter-builder consistency. The epistemic validity of claims produced downstream is governed exclusively by the Alexandria protocol and is not affected by these parameters.

## 8 Claim Candidates

A ClaimCandidate is a structured, pre-protocol object derived from a SemanticProjection that satisfies the emission conditions. It is formally defined as:

> ClaimCandidate {
>
> subject : entity with type annotation
>
> predicate : element of permitted relation family ℛ
>
> object : entity with type annotation
>
> temporal_ordering : AllenRelation ∈ ᴺ\_{Allen} // orthogonal, default: unspecified
>
> category : ClaimCategory
>
> modality : Modality
>
> scope : Scope // population and domain qualifiers only
>
> candidate_score : float // P(r\* \| u), relational confidence
>
> candidate_rank : int // 1 = argmax, 2..k = multi-emission
>
> }

The field temporal_ordering is a first-class property of the ClaimCandidate, not a qualifier within scope. This separation enables the Diff Engine to detect temporal variants of the same proposition: two claims sharing the same CIK (subject_id, relation_id, object_id, scope) but differing in temporal_ordering represent the same ontic relationship asserted under different temporal mechanisms. The Synapse layer can compare such temporal variants as a distinct analysis dimension.

The scope field now exclusively covers population and domain qualifiers (e.g., scope=clinical-trial-2022, scope=general-population). Temporal qualifiers are moved to temporal_ordering. The CIK definition in Section 10.2 remains valid: scope in the hash refers to population/domain scope, which is the correct identity discriminant.

> subject : entity with type annotation
>
> relation : element of permitted relation family
>
> object : entity or property with type annotation
>
> category : EMPIRICAL \| MODEL \| SPECULATIVE \| NORMATIVE
>
> modality : element of M_modal
>
> scope : temporal, population, or domain qualifier
>
> score : P(r \| u) for the selected relation
>
> rank : position among top-k candidates

A ClaimCandidate is not a claim. It has not passed through adjudication, diff, or seal. Its status within the pre-protocol workspace is one of:

> PROJECTED initial state after emission
>
> AMBIGUOUS blocked by Rule E3
>
> READY_FOR_CLAIM passed quality checks, ready for protocol ingestion
>
> BRANCH_CANDIDATE flagged by Rule E4 for dual-builder treatment

Claim candidates carry back-references to their originating SemanticProjection, providing full traceability from canonical claim to source fragment.

## 9 Design Rationale

### 9.1 Probabilistic Semantics vs. Fuzzy Logic

The SPL uses probabilistic representations rather than fuzzy membership functions. This is a principled choice, not a default.

Fuzzy logic models graded truth: a proposition is partially true to degree μ ∈ \[0, 1\]. Probabilistic semantics models uncertainty about interpretation: we are uncertain which of several possible relations is the correct reading of a given text fragment. The two frameworks answer different questions.

In the SPL context, the relevant question is not "how true is this relation" but "which relation does this text fragment most likely express." This is an epistemic uncertainty problem, not a vagueness problem. The Alexandria protocol operates on discrete claims with categorical epistemic status; it requires a projection mechanism that produces discrete candidates from uncertain inputs, not graded truth values.

Additionally, probabilistic representations are directly compatible with the entropy-based ambiguity quantification of Section 7 and with the JSD-based divergence measurement for builder comparison. Fuzzy representations would require a separate formalism for these operations.

### 9.2 Subject-Relation-Object vs. Graph Embeddings

A natural alternative to S-R-O projection is direct graph or text embedding. This alternative is rejected for a structural reason.

Graph and text embeddings encode semantic similarity in a continuous vector space. They are well-suited for tasks such as semantic search, clustering, or paraphrase detection. However, embeddings dissolve epistemic structure: the embedding of "Smoking may cause cancer" and "Smoking causes cancer" are close in standard embedding spaces, yet the epistemic difference between SUGGESTS and CAUSES is fundamental to how a claim should be treated in adjudication.

S-R-O projection forces an explicit relational commitment. This commitment is precisely what the downstream protocol operations of diff, adjudication, and seal require. The SPL does not replace embeddings --- they may be used internally as the computational mechanism for type classification and relation projection --- but the output structure must be a discrete relational form, not a continuous vector.

### 9.3 Pipeline Architecture and Object Types

The SPL transforms natural language into Alexandria-compatible claim structures through a fixed four-stage pipeline. Each stage produces a distinct object type; no stage bypasses its predecessor.

> Source Text
>
> ↓
>
> Semantic Units \[u ∈ U\]
>
> ↓
>
> Semantic Projections \[π ∈ Π\]
>
> ↓
>
> Claim Candidates \[C ∈ Ψ\]
>
> ↓
>
> Alexandria Protocol \[Diff / Adjudication / Seal\]

The pipeline solves a structural mismatch: natural language contains ambiguity; the Alexandria protocol requires discrete, versioned claims. The SPL does not resolve ambiguity --- it makes ambiguity explicit and measurable before it reaches the protocol layer.

A SemanticUnit is the smallest interpreted language fragment that carries a single relational assertion. Its schema:

> SemanticUnit {
>
> semantic_unit_id string
>
> source_ref string // pointer to source document
>
> source_language string
>
> original_text string
>
> context_window string // surrounding text used for typing
>
> created_at timestamp
>
> }

A SemanticProjection represents the full distributional interpretation of a SemanticUnit, including relational, modal, and scope uncertainty:

> SemanticProjection {
>
> semantic_projection_id string
>
> semantic_unit_id string // FK to SemanticUnit
>
> subject_candidates \[\]Entity
>
> relation_distribution {}float // P(r \| u) over permitted relations
>
> object_candidates \[\]Entity
>
> category_distribution {}float
>
> modality_distribution {}float // asserted / suggested / hypothesized\...
>
> scope_candidates {}Scope
>
> uncertainty_profile {}float
>
> projection_entropy float // H_norm
>
> builder_origin string
>
> }

A ClaimCandidate is the output unit of the SPL. It is not yet a canonical claim; it has passed the emission rules but has not been adjudicated or sealed:

> ClaimCandidate {
>
> claim_candidate_id string
>
> semantic_projection_id string // provenance back-reference
>
> subject Entity
>
> predicate Relation
>
> object Entity
>
> category ClaimCategory
>
> modality Modality
>
> scope Scope
>
> qualifiers \[\]Qualifier
>
> candidate_score float // P(r \| u) of selected relation
>
> candidate_rank int // 1 = argmax, 2..k = multi-emission
>
> }

### 9.4 Projection as Tensor Structure

The relational distribution of a SemanticProjection is formally represented as a sparse tensor:

R(i, j, k) = P(subject = i, relation = j, object = k \| u)

where i indexes subject candidates, j indexes relations permitted by M(t_s, t_o), and k indexes object candidates. ClaimCandidates are produced by argmax (E1) or top-k extraction (E2) over this tensor. The tensor is sparse: most (i, j, k) entries are zero because M constrains admissible relations by type-pair before projection.

### 9.5 Entropy Interpretation

The normalised Shannon entropy H_norm quantifies the spread of the relational distribution. Its interpretation is consistent across domains:

  ------------------------- -------------------------------------------------------------------------------
  **H_norm range**          **Interpretation**

  **0.00**                  **Single relation has full probability mass. Unambiguous.**

  **0.00 -- 0.20**          **Low ambiguity. One relation dominates; minor alternatives present.**

  **0.20 -- 0.50**          **Moderate ambiguity. Multiple plausible relations. E2 likely.**

  **0.50 -- 0.65**          **High ambiguity. Borderline E2 / E3. Depends on τ₃ calibration.**

  **≥ 0.65 (default τ₃)**   **Critical ambiguity. E3 block. No emission without revision.**

  **1.00**                  **Maximum uncertainty. All relations equally probable. Fully uninformative.**
  ------------------------- -------------------------------------------------------------------------------

Shannon entropy is chosen over alternatives (Gini impurity, energy scores) because it satisfies the Shannon--Khinchin axioms for uncertainty measures and carries a direct information-theoretic interpretation: H_norm measures the expected informational surprise, not a geometric or energetic proxy.

### 9.6 Relation to Frame Semantics

Frame semantics models event structures: who did what to whom under which circumstances. This is appropriate for event extraction but does not align with the epistemic requirements of Alexandria.

  ---------------------- -----------------------------------------------------------------------
  **Dimension**          **Frame Semantics vs. SPL**

  **Unit of analysis**   **Event frame with roles / S--R--O relational triple**

  **Primary goal**       **Event understanding / Epistemic claim extraction**

  **Output**             **Role-filled frame structure / ClaimCandidate with modality**

  **Diffable**           **No (frames are not versioned) / Yes (claims are diffs over graph)**

  **Adjudicable**        **No / Yes (via Alexandria adjudication rules)**
  ---------------------- -----------------------------------------------------------------------

The critical distinction is modality. A frame analysis of „The data suggest that remote work increases productivity" produces a Suggesting event with embedded proposition. The SPL produces a ClaimCandidate with predicate increases, modality suggested, and score P = f(Π). The modality is a first-class field, not an embedded qualifier --- because it determines whether the claim is diffable against a later ASSERTED version of the same proposition.

### 9.7 Separation of Concerns

The SPL enforces a strict three-way separation that is architecturally non-negotiable:

**Language interpretation:** Handled by the SPL. The SPL determines what the text most plausibly asserts, and with what confidence. It does not determine whether the assertion is true.

**Epistemic structuring:** Handled by the ClaimCandidate schema and emission rules. The structure of a claim (S--R--O, modality, scope) is fixed before it reaches the protocol.

**Protocol-based knowledge validation:** Handled exclusively by the Alexandria protocol (Diff, Adjudication, Seal). Epistemic states --- whether a claim is accepted, disputed, branched, or retracted --- arise only here.

This separation increases stability (a change in the language model does not invalidate sealed claims), language-independence (the same protocol operates on projections from any language), and comparability (claims from different sources are structurally comparable because they share the same S--R--O schema).

## 10 Integration with the Alexandria Protocol

The SPL is integrated into the Alexandria Protocol as a pre-protocol stage. The following invariants govern this integration:

-   Protocol invariance: the SPL does not modify any Alexandria protocol object, operation, or rule.

-   No direct text-to-claim: no natural language segment may be ingested as a canonical claim without passing through the SPL or a formally equivalent interpretation layer.

-   Candidate provenance: every canonical claim carries a reference to its originating SemanticProjection.

-   Configuration independence: changes to Θ do not affect the protocol layer. Two implementations with different threshold sets may produce different claim candidates from the same source; any resulting claim divergence is handled by the protocol's standard diff and adjudication mechanisms.

The integration point between SPL and protocol is the ClaimCandidate in state READY_FOR_CLAIM. Beyond this point, the Alexandria protocol assumes full responsibility for epistemic validation.

### 10.2 Diff-Equivalence Rule and Canonical Identity Key (CIK)

For the Alexandria protocol to compute a Diff over probabilistic SPL outputs, it must distinguish between the propositional identity of a claim and its epistemic state. Without this distinction, two ClaimCandidates with the same subject-relation-object triple but different candidate_scores would be treated as different propositions, causing duplicate ingestion.

When a ClaimCandidate in state READY_FOR_CLAIM enters the protocol layer, the system computes its Canonical Identity Key (CIK):

> CIK = hash(subject_id, relation_id, object_id, scope)

The Diff Engine operates strictly on the CIK. The candidate_score (the relational probability P(r\*\|u) from the SPL) is attached as evidence metadata but is not part of the identity computation. A claim is the same proposition regardless of whether the builder was 81% or 82% confident in it.

Three cases govern CIK-based Diff processing:

**Insertion:** The CIK does not exist in the target BranchNode. The candidate is treated as a new proposition and ingested as an ADD operation. Status transitions to PROJECTED within the protocol layer.

**Epistemic Update:** The CIK exists, but the incoming candidate carries a different modality (e.g., the existing sealed claim has modality asserted, and the new candidate has modality disputed). This is processed as a modality-state patch, not a new claim. The candidate_score is recorded in the patch audit log.

**Adjudication Trigger:** Two builders submit candidates with the same CIK simultaneously, but with contradictory modalities (e.g., supported vs. refuted). The Diff Engine blocks automatic merging and generates an AdjudicationTicket. The candidate_scores from both builders are attached as evidence for adjudicators, but do not determine the outcome.

*Scope is included in the CIK because the same subject-relation-object triple with different scopes represents different propositions. Example: (RemoteWork, increases, Productivity, scope=2020-pandemic) and (RemoteWork, increases, Productivity, scope=general) are distinct CIKs and are stored as distinct claims. A Diff between them is an ADD, not an epistemic update.*

The CIK definition resolves a structural ambiguity in the SPL-to-protocol interface: the probabilistic outputs of the SPL are position-agnostic (they describe what a text most plausibly asserts), while the protocol is identity-based (it tracks the evolution of specific propositions). The CIK is the translation layer between these two modes.

## 11 Relation to Existing Work

The SPL draws on and extends several existing frameworks in knowledge representation and natural language processing.

**RDF Schema and OWL.** The typed relation matrix is structurally similar to property domain and range constraints in RDF Schema and OWL. The SPL differs in that relation assignment is probabilistic rather than deterministic: the matrix constrains the candidate space, but the projection computes distributions over that space rather than making binary admissibility decisions.

**Frame Semantics.** The SemanticUnit and its associated role structure (subject, relation, object, modality, scope) are related to the concept of frames in Fillmore's frame semantics. The SPL formalizes this structure with explicit probabilistic semantics and a constrained relation vocabulary.

**Information Extraction.** Relation extraction in the information extraction literature typically produces deterministic (subject, relation, object) triples. The SPL produces distributions over candidate triples, preserving uncertainty for downstream protocol processing rather than discarding it at extraction time.

**Typed Knowledge Graphs.** Systems such as Freebase, Wikidata, and biomedical knowledge graphs employ typed entities and typed relations. The SPL adopts this approach and extends it with probabilistic projection and entropy-based ambiguity quantification.

## 12 Open Questions

Several design questions are left open in this draft and will be addressed in subsequent versions.

-   Temporal granularity in ᴺ\_{Allen}. Section 6.3 introduces the orthogonal temporal dimension τ via Allen's Interval Algebra. The open question is the appropriate granularity: the full 13-relation algebra, the reduced 8-relation set used here, or a domain-specific subset. Empirical calibration on annotated corpora is required to determine which Allen relations are linguistically recoverable from natural language with sufficient precision to be actionable in P(τ \| u).

-   Matrix governance. The typed relation matrix is a critical design artifact: changes to M affect what claims are structurally expressible. A formal versioning and governance mechanism for M is required, analogous to the patch and seal mechanism of the protocol layer.

-   Calibration methodology. The recommended threshold values in Θ are based on theoretical constraints and informal reasoning. A systematic empirical calibration against annotated scientific corpora is needed to establish evidence-based default values.

-   Semantic distance for Synapse. The projection tensor provides a natural basis for computing semantic distance between claim graphs, which is required for the Synapse layer. The formal definition of this distance measure is deferred to a subsequent paper.

## 13 Conclusion

This paper has presented the Semantic Projection Layer, a formally defined pre-protocol stage for the Alexandria Protocol. The SPL addresses the structural gap between natural language and the discrete claim objects required by the protocol.

The SPL decomposes source text into minimal epistemic units, assigns type classes to their constituents, constrains the relational search space via a typed relation matrix, and computes probabilistic projections over the constrained space. Ambiguity is quantified by Shannon entropy; builder divergence by Jensen-Shannon divergence. Emission rules derive structured claim candidates from the projection.

The SPL is pre-protocolar: it is interpretive, not normative. It generates candidate structures for the protocol to evaluate. The epistemic validity of claims remains fully under the governance of the Alexandria protocol.

The principal contribution of the SPL is the establishment of a principled, language-independent, formally defined bridge between natural language interpretation and epistemic versioning. This bridge is a prerequisite for scaling the Alexandria Protocol to multilingual, multi-domain scientific knowledge.

*Alexandria Protocol Working Paper Series*

Draft v0.1 --- Subject to revision

## Appendix H --- Module Reference: Semantic Projection Layer

*This appendix summarises the Semantic Projection Layer as an integral component of the Alexandria v2.2 architecture. It serves as the technical reference sheet for module finalisation and implementation.*

### H.1 Purpose and Definition

The SPL functions as the formal bridge (Layer 2) between unstructured natural language and the highly structured Alexandria protocol. Its primary task is to **mathematically quantify** the interpretive ambiguity present in the transformation of text into relational claims, rather than suppressing it. The SPL is pre-protocolar: it produces ClaimCandidates, not sealed claims. Epistemic validity remains under the exclusive governance of the Alexandria protocol.

### H.2 The Operational Pipeline

Information processing in the SPL follows a three-stage filter cascade.

**Stage A --- Structural Validation (Typed Relation Matrix)**

**Mechanism:** Each extracted SemanticUnit is checked against the Typed Relation Matrix M.

**Logic:** Relations are admissible only when the type combination of subject and object (e.g. CHEMICAL → BIOLOGICAL_PROCESS) permits the corresponding relation family (e.g. DYNAMIC).

**Goal:** Prevention of categorical errors, in particular the conflation of statistical correlation with causal mechanism.

**Stage B --- Ambiguity Measurement (Shannon Entropy)**

**Mechanism:** Computation of normalised Shannon entropy H~norm~ over the relational distribution P~r~.

> H(P_r) = - Σ p_i · log(p_i)

**Interpretation:** A high entropy value signals an unclear source or vague language. This triggers emission blockage under Rule E3.

**Stage C --- Divergence Check (Jensen-Shannon Divergence)**

**Mechanism:** Comparison of the interpretation distributions of different builders using JSD.

**Logic:** When divergence exceeds threshold τ~4~, an epistemic conflict is detected.

**Result:** Automatic generation of a BRANCH_CANDIDATE to structurally represent the dissent in the graph.

### H.3 Emission Ruleset (E1--E4)

The decision whether information enters the protocol follows strict threshold rules governed by the configuration parameter set Θ = {τ~1~, τ~2~, τ~3~, τ~4~}.

  ---------- ----------- --------------------------------------- ----------------------------------------------
  **Rule**   **Label**   **Condition**                           **System Action**

  **E1**     Singular    High dominance, low entropy             Emits single ClaimCandidate (argmax)

  **E2**     Multiple    Low dominance, moderate entropy         Emits top-k ClaimCandidates for adjudication

  **E3**     Block       Critical entropy (H~norm~ ≥ τ~3~)       Information discarded; status = AMBIGUOUS

  **E4**     Branch      High builder divergence (JSD \> τ~4~)   Creates graph fork (BRANCH_CANDIDATE)
  ---------- ----------- --------------------------------------- ----------------------------------------------

### H.4 Integration in Alexandria v2.2

The SPL is pre-protocolar. It produces ClaimCandidates, not final claims. Only after passing through the SPL are data ingested into the canonical protocol layer (Diff, Adjudication, Seal). Each claim carries a permanent reference to its originating semantic projection, providing complete traceability from sealed claim back to source fragment.

> Source Text
>
> ↓
>
> **Semantic Projection Layer \[SPL\]**
>
> ↓ Matrix → Entropy → JSD → E1/E2/E3/E4
>
> ClaimCandidates
>
> ↓
>
> Alexandria Protocol \[Diff / Adjudication / Seal\]

*The Semantic Projection Layer is formally specified and ready for implementation as a component of Alexandria v2.2.*

## Appendix I --- SPL Operations Manual & Implementation Reference

*This appendix documents the operational logic of the SPL in executable detail: emission decision paths, worked examples, builder divergence simulation, technical module specifications, data schemas, and reference implementation code.*

### I.1 Emission Logic and Decision Paths

Complete emission logic governed by Θ = {τ₁--τ₄}. Recommended initial values: τ₁ ≈ 0.60, τ₂ ≈ 0.25, τ₃ ≈ 0.65, τ₄ ≈ 0.40.

  ---------- ----------- --------------------------------- ------------------ -------------
  **Rule**   **Label**   **Condition**                     **Status**         **Action**

  **E1**     Singular    max(P_r) \> τ₁ AND H_norm \< τ₂   READY_FOR_CLAIM    Emit argmax

  **E2**     Multiple    max(P_r) ≤ τ₁ AND H_norm \< τ₃    READY_FOR_CLAIM    Emit top-k

  **E3**     Block       H_norm ≥ τ₃                       AMBIGUOUS          No emission

  **E4**     Branch      JSD(Pᴬ, Pᴭ) \> τ₄                 BRANCH_CANDIDATE   Fork graph
  ---------- ----------- --------------------------------- ------------------ -------------

### I.2 Worked Example: AMBIGUOUS Block (Rule E3)

**Source:** *„Die Ergebnisse deuten darauf hin, dass Wirkstoff X die Protein-Interaktion Y hemmen könnte, wobei ein direkter kausaler Beleg noch aussteht."*

> u2: WirkstoffX → koennte_hemmen → ProteinInteraktionY
>
> M(CHEMICAL, BIOLOGICAL_PROCESS) → {DYNAMIC, STATISTICAL}
>
> Distribution: inhibits=0.48 associates_with=0.32 decreases=0.15 affects=0.05
>
> Modality: suggested (P=0.85) \[signal: „könnte"\]
>
> max(P_r)=0.48 H_norm=0.72
>
> E1: max(P_r) \> τ₁(0.60)? NO. E3: H_norm ≥ τ₃(0.65)? YES → AMBIGUOUS

No emission. The hedged formulation („deuten darauf hin", „könnte", „steht aus") produces sufficient distributional spread that no relation achieves stable dominance. The SPL correctly refuses to emit a hard causal claim. This is the intended protective behaviour: soft scientific hypotheses do not enter the graph as structural facts.

### I.3 Builder Divergence Simulation (Rule E4)

Builder A reads a causal inhibition (inhibits=0.70); Builder B reads a statistical association (associates_with=0.70).

> P = \[0.70, 0.10, 0.20\] Q = \[0.10, 0.70, 0.20\] M = \[0.40, 0.40, 0.20\]
>
> KL(P\|\|M) ≈ 0.32 KL(Q\|\|M) ≈ 0.32 JSD = 0.32

E4: JSD(0.32) \> τ₄(0.40)? **No.** Disagreement is within tolerance → E2 (multiple emission). At JSD \> 0.40 the system would create a BRANCH_CANDIDATE and record the epistemic conflict as a graph fork for downstream adjudication. Contradictions are not averaged out; they are made structurally visible.

### I.4 Technical Module Specification

**M1 --- Epistemic Fragmentation.** Segments text into SemanticUnits via boundary signals. Generates (subject, relation, object) triples. Single sentences must be decomposable into multiple units.

**M2 --- Distributional Type System.** Computes P_T(type\|token,context). Tokens with type entropy above threshold flagged AMBIGUOUS_TYPE and routed to dual-builder processing.

**M3 --- Relational Filter and Projection.** Implements M. Enforces STATISTICAL/DYNAMIC separation. Two-stage projection: P(f\|u) then P(r\|f,u).

**M4 --- Ambiguity and Divergence Engine.** Computes H_norm and JSD. Manages configurable Θ.

**M5 --- Emission and Protocol Interface.** Applies E1--E4. Generates ClaimCandidate objects with score, modality, scope. Every candidate carries a back-reference to its originating SemanticProjection.

### I.5 Typed Relation Matrix (JSON-LD)

> { "@context": {"alexandria": "https://protocol.alexandria.org/v2.2/vocab#"},
>
> "@type": "alexandria:TypedRelationMatrix", "version": "2.2.0-SML-Draft",
>
> "rules": \[
>
> {"subjectType":"alexandria:CHEMICAL","objectType":"alexandria:BIOLOGICAL_PROCESS",
>
> "permittedFamilies":\["alexandria:DYNAMIC","alexandria:STATISTICAL"\]},
>
> {"subjectType":"alexandria:ENTITY","objectType":"alexandria:PROPERTY",
>
> "permittedFamilies":\["alexandria:ONTIC"\]},
>
> {"subjectType":"alexandria:CLAIM","objectType":"alexandria:CLAIM",
>
> "permittedFamilies":\["alexandria:EPISTEMIC"\]} \] }

### I.6 Reference Implementation (Python)

> import json, math
>
> class SPLFullEngine:
>
> def \_\_init\_\_(self, matrix_path, thresholds):
>
> with open(matrix_path) as f: self.rules = json.load(f).get("rules",\[\])
>
> self.tau = thresholds
>
> def validate(self, s_type, o_type, family):
>
> for r in self.rules:
>
> if r\["subjectType"\]==f"alexandria:{s_type}" and r\["objectType"\]==f"alexandria:{o_type}":
>
> return f"alexandria:{family}" in r\["permittedFamilies"\]
>
> return False
>
> def entropy(self, dist):
>
> h = -sum(p\*math.log2(p) for p in dist if p\>0)
>
> return h/math.log2(len(dist)) if len(dist)\>1 else 0.0
>
> def emit(self, dist):
>
> p, h = max(dist), self.entropy(dist)
>
> if p\>self.tau\['tau1'\] and h\<self.tau\['tau2'\]: return "E1:READY(singular)",h
>
> if h\>=self.tau\['tau3'\]: return "E3:AMBIGUOUS",h
>
> return "E2:READY(multiple)",h
>
> def \_kl(self,p,q): return sum(p\[i\]\*math.log2(p\[i\]/q\[i\]) for i in range(len(p)) if p\[i\]\>0 and q\[i\]\>0)
>
> def compare_builders(self, p, q):
>
> m=\[0.5\*(p\[i\]+q\[i\]) for i in range(len(p))\]
>
> jsd=0.5\*self.\_kl(p,m)+0.5\*self.\_kl(q,m)
>
> return ("E4:BRANCH_CANDIDATE" if jsd\>self.tau\['tau4'\] else "CONSENSUS_STABLE"), jsd
>
> \# Usage
>
> theta={'tau1':0.60,'tau2':0.25,'tau3':0.65,'tau4':0.40}
>
> engine=SPLFullEngine('matrix.jsonld', theta)
>
> print(engine.emit(\[0.48,0.32,0.15,0.05\])) \# E3:AMBIGUOUS
>
> print(engine.compare_builders(\[.70,.10,.20\],\[.10,.70,.20\])) \# CONSENSUS_STABLE

*SPL module complete. All three filter stages implemented: structural validation (Matrix), ambiguity measurement (Entropy / E1--E3), divergence detection (JSD / E4). Ready for integration into Alexandria v2.2.*

## Appendix J --- Self-Referential Matrix Governance

*The Typed Relation Matrix M is not treated as a privileged static configuration. Instead, M is itself subject to the Alexandria protocol: matrix rules are represented as Meta-Claims, and all changes to M must pass through the same audit, adjudication, and sealing cycle as ordinary claims. This eliminates the last structural single point of failure --- the human administrator who sets the rules --- and makes the evolution of M fully transparent and auditable.*

### J.1 Core Principle: Matrix Rules as Meta-Claims

A rule in M of the form „M(CHEMICAL, BIOLOGICAL_PROCESS) → {DYNAMIC, STATISTICAL}" is a structural assertion about the knowledge system itself. It is treated as a **Meta-Claim** of category NORMATIVE with relation permits. Its subject is a type-pair; its object is a relation family. Changes to M are submitted as Patch objects and processed through the full protocol cycle before becoming operative.

> MetaClaim: (CHEMICAL × BIOLOGICAL_PROCESS) ---permits→ DYNAMIC
>
> Category: NORMATIVE \| Status: SEALED \| BranchNode: M_v2.2.0

A new type-relation pairing is a Patch of type ADD (Regel 3: extension). A modification to an existing pairing is a Patch of type REPLACE (Regel 2: supersession). Both require the same three-level audit before enactment.

### J.2 Three-Level Audit for Matrix Changes

**Level 1 --- Patch Audit (syntactic).** Is the proposed matrix rule syntactically valid JSON-LD? Do subject type and object type exist in the type system T? Is the proposed relation family one of the six canonical families? This level is fully automated and deterministic.

**Level 2 --- Claim Audit (semantic).** Does the proposed rule contradict any already-sealed Meta-Claim in the current matrix version? This is a direct conflict check against the sealed claim set. A rule that would simultaneously permit and forbid the same relation for the same type-pair is rejected at this level. Result: PASS or CONFLICT (with pointer to the conflicting sealed claim).

**Level 3 --- Graph Audit (structural impact).** Would the new rule destabilise existing claims in the knowledge graph? This audit operates on a **bounded local neighbourhood** of the affected type-pair, not on the full graph. Global consistency is not guaranteed --- and this constraint is explicit (see J.5). The audit reports a MappingConfidenceDelta: the change in relational confidence scores across all existing claims whose type-pairs are affected by the proposed rule.

### J.3 Governance Workflow

  ------------------ ------------------------------------------------------------------------------------------------------------------------------------------------------ ------------------------------
  **Phase**          **System Action**                                                                                                                                      **Result**

  **Proposal**       Submission of a new type-relation pairing as a Patch object (ADD or REPLACE).                                                                          Status: PROJECTED

  **Verification**   Three-level audit (L1 syntax, L2 conflict, L3 local impact). MappingConfidenceDelta computed.                                                          Audit log with impact score

  **Consensus**      JSD comparison across builder interpretations of the proposed rule. No alpha-default: acceptance requires explicit adjudication.                       ACCEPTED or BRANCH_CANDIDATE

  **Enactment**      Sealing of new matrix version via Seal module. New BranchNode M_v{n+1} created. Previous version M_v{n} remains accessible for retrospective audits.   Status: SEALED
  ------------------ ------------------------------------------------------------------------------------------------------------------------------------------------------ ------------------------------

### J.4 Epistemic Branching via Matrix Versioning

Each sealed matrix version M_v{n} is bound to a BranchNode in the knowledge graph. This enables parallel processing of knowledge corpora under different matrix versions --- a property termed **Epistemic Branching**. A claim sealed under M_v2 and a claim sealed under M_v3 are not directly comparable without explicit inter-branch adjudication, because they may have been validated against different structural rules.

This is not a weakness. The matrix version is part of the provenance chain of every sealed claim. Downstream consumers of the graph can query which matrix version a claim was validated against, and decide whether to accept, re-validate, or branch accordingly. The alternative --- silently migrating all existing claims to a new matrix --- would destroy the audit trail.

### J.5 Open Problems and Explicit Constraints

**1. The Bootstrapping Problem.** Self-referential governance requires a starting point. The genesis version M_v1.0 cannot be validated by the protocol, because the protocol requires M to operate. M_v1.0 is therefore a **declared axiom** --- set by the protocol designers outside the governance cycle, carrying the status AXIOM rather than SEALED. All subsequent versions M_v{n+1} are governed. This constraint is not eliminable; it is the system's Gödel floor and is documented as such.

**2. Level-3 Audit Is Bounded, Not Complete.** Global consistency checking --- whether a new matrix rule creates logical paradoxes anywhere in the graph --- is undecidable in the general case for sufficiently large graphs. The Level-3 audit therefore operates on a configurable local neighbourhood (default: depth 3 from affected type-pairs). This provides practical protection against local structural breakage while acknowledging that full global guarantees are computationally intractable. The audit scope parameter is itself a governance-controlled variable in Θ.

**3. Inter-Branch Adjudication Protocol.** Claims from different matrix branches are not directly comparable. An inter-branch adjudication rule is required to merge or compare them. This rule is not defined in the current version and is designated as an open problem for v2.3. Until resolved, cross-branch comparisons must be flagged as REQUIRES_INTER_BRANCH_ADJUDICATION in query results.

### J.6 Architectural Coherence

Matrix governance via the protocol eliminates the privileged administrator role for all versions after M_v1.0. It does not eliminate the need for human judgment --- builders who propose and adjudicate matrix changes are human or AI agents operating under the same rules as for ordinary claims. What changes is that their decisions are no longer opaque: every matrix change has a full audit trail, a JSD-based consensus score, and a sealed provenance record. The governance of the structure that governs the knowledge becomes itself governable knowledge.

*The bootstrapping problem (J.5.1), the bounded audit constraint (J.5.2), and the open inter-branch adjudication problem (J.5.3) are not provisional gaps to be papered over. They are structural features of any self-referential formal system and are documented here as part of the specification rather than in spite of it.*

### J.7 Type System Governance and Append-Only Evolution

The foundational type set T (Section 4.1) is subject to the same self-referential governance as the Typed Relation Matrix M. The existence of a type class is represented in the system as an ontic Meta-Claim:

> MetaClaim: (Algorithm) ---instance_of→ (Alexandria:TypeClass)
>
> Category: ONTIC \| Status: SEALED \| BranchNode: T_v1.1.0

**Append-Only Constraint.** Unlike matrix rules which can be replaced (RULE_2_REPLACE), the type system vocabulary operates under a strict append-only constraint. A new type can be added via a TypePatch (RULE_3_EXTEND semantics), but once a type is sealed into the vocabulary, it cannot be deleted or renamed.

Removing a type would orphan all historical claims that use that type, breaking the immutability guarantee of prior BranchNodes. Deprecation of a type is achieved structurally: builders agree via MatrixPatch operations to remove all permitted relations for the deprecated type in M, effectively isolating it as a node with no permitted relations. Historical records remain intact; the type simply becomes unreachable for new claims.

**Two-Step Bootstrapping of a New Type.** When a new type (e.g., Algorithm) is sealed via a TypePatch, it enters the vocabulary with an empty relational matrix: M(Algorithm, X) = ∅ for all X ∈ T. The new type cannot be used in any SemanticUnit until subsequent MatrixPatch operations establish its permitted relation families. This two-step initialisation prevents unconstrained semantic sprawl: a type exists in the vocabulary before it can generate claims.

Example: Adding Algorithm to T.

> • Step 1 (TypePatch): Seal MetaClaim (Algorithm) ---instance_of→ (Alexandria:TypeClass). T is now T_v1.1.0. Algorithm is in the vocabulary but has no permitted relations.
>
> • Step 2 (MatrixPatch): Propose M(Algorithm, Process) → {DYNAMIC}. This patch passes the three-level audit (J.2). On sealing, Algorithm can now participate in DYNAMIC relations with Process entities.

*STRUCTURAL_VIOLATION signals (Rule E0, Section 7.2) are a data-driven input to this process. Persistent violations for a type-pair indicate that the corpus contains assertions the current M cannot accommodate. If the violations reflect legitimate scientific usage rather than source errors, a MatrixPatch is warranted. If the type itself does not exist in T, a TypePatch is required first.*

**Governance Workflow for TypePatch.** Identical to the MatrixPatch workflow (J.3), with one additional check at Level 2: does the proposed type name conflict with any existing type or known ontological class in T? Name conflicts are rejected at L1 (syntax). Semantic overlaps (e.g. proposing Dataset when Model already covers the use case) are flagged at L2 as CLAIM_CONFLICT and require explicit adjudication before the TypePatch can be enacted.

## Appendix K --- Context-Aware BranchNode & Matrix Migration

*The original BranchNode (v2.2) was a container for claims and metadata. This appendix specifies the extended schema that makes each BranchNode self-describing with respect to the matrix under which its claims were validated. The consequence: the architecture becomes immutable. A branch without a matrix reference is epistemically uninterpretable, since the meaning of its relations cannot be reconstructed.*

### K.1 Extended BranchNode Schema

Each BranchNode now carries a structural_context field that records both the vocabulary version (T) and the matrix version (M) active when its claims were validated, and a cryptographic seal of that version. This makes the interpretive grammar of the branch permanently retrievable.

> {
>
> "branch_id": "BN-2026-07-alpha",
>
> "parent_branch": "BN-2026-06-main",
>
> "structural_context": {
>
> "vocabulary_version": "v1.1.0-Types",
>
> "vocabulary_seal_hash": "sha256:abcd1234ef56\...",
>
> "matrix_version": "v2.2.0-SML",
>
> "matrix_seal_hash": "sha256:e3b0c44298fc\..."
>
> "stringency_profile": {
>
> "alpha": 0.30,
>
> "beta": 0.20,
>
> "gamma": 0.20,
>
> "delta": 0.30,
>
> "tau_seal": 0.75
>
> }
>
> },
>
> "adjudication_strategy": "jsd_threshold_tau4",
>
> "claims": \[
>
> { "claim_id": "C-0042", "status": "SEALED", \... }
>
> \]
>
> }

**Field notes:** matrix_version identifies the sealed M version active at claim-time. matrix_seal_hash is the cryptographic fingerprint of the sealed JSON-LD matrix rules --- not of the claims. Even if the global matrix configuration is subsequently altered, any branch remains verifiable against its original hash.

### K.2 DiffNode Logic: Detecting Structural Matrix Mismatch

When two BranchNodes are compared via a DiffNode, the system first checks whether their matrix contexts are identical. This determines whether a divergence in claims reflects a difference in data or a difference in the interpretive grammar itself.

  ---------------------- ------------------------------------------ ------------------------------------------------------------------------------------------------------------------------
  **Comparison**         **Condition**                              **System Action**

  **Same matrix**        **matrix_version A == matrix_version B**   **Standard semantic diff proceeds.**

  **Different matrix**   **matrix_version A ≠ matrix_version B**    **DiffNode receives metadata reason: structural_matrix_mismatch. Diff is flagged REQUIRES_INTER_BRANCH_ADJUDICATION.**

  **Hash conflict**      **Versions match but hashes differ**       **Integrity violation. Branch flagged INTEGRITY_FAILURE. No diff proceeds.**
  ---------------------- ------------------------------------------ ------------------------------------------------------------------------------------------------------------------------

The reason: structural_matrix_mismatch flag is not an error condition --- it is an accurate description. Two claims validated under different matrices are not comparable without an explicit inter-branch adjudication rule (open problem, documented in J.5.3). The flag prevents silent conflation of semantically incomparable claims.

### K.3 Matrix Migration Workflow

Updating M from version v{n} to v{n+1} is a protocol-governed event, not a configuration change. The following four steps constitute a Matrix Migration:

**Step 1 --- Patch Creation**

A Matrix Patch object is created specifying the target matrix version, the author's EpistemicIdentity, and the proposed modifications as structured rule objects (see K.4 for JSON-LD example). The patch carries a checksum of its own content.

**Step 2 --- Governance Audit**

The patch passes through the three-level audit defined in J.2: syntactic validity (L1), conflict with sealed Meta-Claims (L2), and local graph impact via MappingConfidenceDelta (L3). Builder JSD comparison determines consensus. If JSD exceeds τ₄, the patch itself becomes a BRANCH_CANDIDATE rather than being accepted into the main branch.

**Step 3 --- New BranchNode Creation**

On acceptance, a new BranchNode is created carrying the new matrix_context (v{n+1} with updated seal hash). The previous BranchNode (v{n}) is not deleted --- it remains accessible for retrospective audits and for branches that were sealed under the old matrix.

**Step 4 --- Optional Re-Projection**

If the new matrix rule affects the relational interpretation of existing claims, those claims can be re-projected through the SPL under the new M. Re-projection is optional and must be explicitly requested. Re-projected claims receive a new ClaimCandidate status and must pass through the full emission and sealing cycle again. They do not automatically supersede the originals.

### K.4 Matrix Patch: JSON-LD Example

The following patch introduces the relation regulates (DYNAMIC family) for the CHEMICAL → BIOLOGICAL_PROCESS type pair, replacing the previous rule R-CHEM-BIO-01 (RULE_2_REPLACE semantics).

> {
>
> "@context": {
>
> "alexandria": "https://protocol.alexandria.org/v2.2/vocab#",
>
> "patch": "alexandria:MatrixPatch"
>
> },
>
> "@type": "patch",
>
> "patch_id": "MP-2026-004",
>
> "target_matrix_version":"v2.2.0-SML",
>
> "author": {
>
> "@type": "alexandria:EpistemicIdentity",
>
> "id": "builder_alpha_01"
>
> },
>
> "modifications": \[
>
> {
>
> "rule_id": "R-CHEM-BIO-01",
>
> "semantics": "RULE_2_REPLACE",
>
> "subjectType": "alexandria:CHEMICAL",
>
> "objectType": "alexandria:BIOLOGICAL_PROCESS",
>
> "addedFamilies": \["alexandria:DYNAMIC"\],
>
> "addedRelations": \["alexandria:regulates"\],
>
> "rationale": "High entropy in inhibits vs activates distribution
>
> suggests need for a neutral regulator term."
>
> }
>
> \],
>
> "checksum": "sha256:e3b0c44298fc1c149afbf4c8996fb92427ae41e4\..."
>
> }

**Patch semantics:** RULE_2_REPLACE supersedes an existing rule for the specified type-pair within the target branch. A completely new type-pair would use RULE_3_EXTEND semantics, which extends the architecture rather than modifying it. Both require the same governance cycle.

### K.5 Audit Pipeline for Matrix Patches

A submitted patch traverses the following verification sequence:

  ------------------- ------------------------------------------------------------------------------------- ------------------------------------------------------------------------
  **Check**           **What is verified**                                                                  **Failure action**

  **Lineage**         **EpistemicIdentity of author is known and not revoked.**                             **Patch rejected: UNKNOWN_IDENTITY.**

  **Checksum**        **patch checksum matches content hash.**                                              **Patch rejected: INTEGRITY_FAILURE.**

  **L1 Syntax**       **JSON-LD valid; types and families exist in T.**                                     **Patch rejected: SYNTAX_ERROR.**

  **L2 Conflict**     **No conflict with sealed Meta-Claims in current M.**                                 **Patch rejected: CLAIM_CONFLICT (with pointer).**

  **L3 Impact**       **MappingConfidenceDelta within acceptable bounds (local neighbourhood, depth 3).**   **Patch flagged: HIGH_IMPACT. Requires elevated adjudication quorum.**

  **JSD Consensus**   **Builder divergence on proposed rule ≤ τ₄.**                                         **If JSD \> τ₄: patch becomes BRANCH_CANDIDATE, not enacted on main.**

  **Enactment**       **All prior checks passed. Seal applied. New BranchNode created.**                    **Status: SEALED. Matrix version incremented.**
  ------------------- ------------------------------------------------------------------------------------- ------------------------------------------------------------------------

### K.6 Immutability Guarantee

The matrix_seal_hash field on each BranchNode provides a cryptographic guarantee that is independent of the current state of the global matrix configuration. Even if M is subsequently modified --- whether through legitimate governance or adversarial action --- the original matrix version that governed a branch can always be reconstructed from its hash and the version history.

This property has a direct consequence for trust: a consumer of a sealed claim does not need to trust the current system configuration. They need only verify that the claim's BranchNode hash corresponds to a known sealed matrix version. The chain of trust runs from claim → BranchNode → matrix_seal_hash → sealed Meta-Claim archive.

*Merksatz: A BranchNode without a matrix_seal_hash is epistemically uninterpretable. The meaning of its relational claims cannot be reconstructed without knowing which structural rules were in force at the time of sealing.*

### K.7 System Closure

The architecture described across Appendices H through K constitutes a closed epistemic system:

> • Natural language is transformed into relational structures via the SPL (Appendix H, I).
>
> • Interpretive ambiguity is quantified and governed by entropy and JSD thresholds (Appendix I).
>
> • The structural rules of interpretation (Matrix M) are themselves subject to the same protocol as the claims they govern (Appendix J).
>
> • Every branch of the knowledge graph carries a cryptographic reference to the specific matrix version under which it was validated (Appendix K).
>
> • Matrix updates are protocol-governed events with full audit trails, consensus requirements, and immutable versioning.

The only element outside this cycle is M_v1.0, the genesis matrix, which is an explicit axiom (documented in J.5.1). All subsequent evolution of the system --- including evolution of its own interpretive grammar --- is governed, auditable, and cryptographically anchored.
