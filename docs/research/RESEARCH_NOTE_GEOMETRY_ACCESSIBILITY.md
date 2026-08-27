# Geometry and Accessibility of Predictive Information in Learned Representations

## A research note on partially observed stochastic dynamical systems

**Working note — 29 July 2026**

**Status.** This document states an empirical phenomenon, introduces a
candidate mathematical formulation, and proposes a controlled research
programme. It does not claim that the proposed mechanism has already been
identified, nor that its novelty relative to the full literature has been
established.

## Abstract

Predictive representations are usually evaluated by asking how accurately a
downstream target can be decoded. This leaves open a different question: how is
predictive information arranged inside the representation, and what resources
are required to recover it? In a controlled study of supervised and
joint-embedding predictive objectives on multivariate sequential data, we find
that predictive content, pooling, and variance-ordered accessibility can
separate sharply. A supervised representation aligns a directional target with
its leading covariance directions. A horizon-predictive representation retains
substantial nonlinear content at the final time step, yet its leading principal
subspace is less predictive than random subspaces of the same dimension, and
temporal averaging selectively removes much of its decodable content. A fixed
all-ones projection over the four token roles also retains much less linear
performance for horizon-JEPA than for supervised. A completed structured
role-Haar diagnostic, however, does not find that this direction or its matched
complement is exceptional across all encoder seeds. The fixed projection is
therefore retained as an operational readout fact, not as evidence for a
privileged relational mechanism. Both representations are strongly
anisotropic, so anisotropy alone does not explain the difference.

We formulate four distinct objects—predictive content, resource-constrained
accessibility, variance–task alignment, and decoder co-adaptation—and emphasize
that a decoded predictor does not generally identify a unique internal
geometry. A minimal example shows that an invertible rescaling can preserve
optimal predictive risk while changing top-PCA accessibility from complete to
null. This motivates a mechanistic replication using partially observed
stochastic processes with known sufficient state, nuisance variables, and
Bayes risk. The proposed programme asks whether training objectives
systematically select different geometries within functionally comparable
representations, and whether variance/covariance or isotropic-Gaussian
regularization can improve finite-sample accessibility without adding or
destroying predictive content.

---

## 1. Empirical phenomenon and general question

The motivating experiment compares three encoders with a substantially shared
token-preserving backbone and matched data splits:

1. an end-to-end supervised encoder;
2. a horizon-JEPA encoder trained to predict future representations
   [[7](#references)];
3. a masked-JEPA encoder trained to complete masked regions.

The empirical system is a Limit Order Book dataset, but the phenomenon of
interest is not specific to market microstructure. Each input is a multivariate
sequence represented on a grid of \(K=20\) time positions, \(S=4\) token roles,
and embedding dimension \(d=128\). The primary readout concatenates the four
tokens at the final position, giving \(D=512\).

Let \(U_m\) denote the first \(m\) principal directions of this readout. At the
common rank \(m/D=1/32\), the fraction of full-rank linear predictive
performance recovered by horizon-JEPA is

\[
\frac{R^2(U_m^\top Z)}{R^2(Z)}=0.0050.
\]

A numerical null based on random orthonormal subspaces of the same dimension
gives \(0.0563\). The corresponding supervised values are \(0.8971\) and
\(0.6118\). Thus, under the measured protocol, the leading variance directions
of horizon-JEPA are systematically less useful for the directional target than
random directions. We call this **variance–task anti-alignment**; it is an
empirical description, not a claim that PCA was theoretically required to
outperform the null.

This effect is not explained by a simple difference in anisotropy. The
participation ratios of the final readouts are \(7.91\) for horizon-JEPA and
\(7.05\) for supervised, despite their opposite accessibility profiles. Nor is
it equivalent to absence of predictive content: the historical post-P0
multiseed nonlinear diagnostic recovers \(R^2=0.3191\) from horizon-JEPA and
\(0.3881\) from supervised at the last position. This diagnostic predates and
is not interchangeable with the later Phase-III-R reader protocol.

The resulting question is:

> How do supervised and self-supervised predictive objectives preserve,
> allocate, and expose predictive variables in representations of partially
> observed stochastic dynamical systems?

More specifically, we want to separate:

\[
\text{predictive content}
\quad\neq\quad
\text{content surviving a chosen pooling}
\quad\neq\quad
\text{content accessible with finite resources}.
\]

---

## 2. Formal setup

Let \((X_t)_{t\in\mathbb Z}\) be a stochastic process with latent decomposition

\[
X_t=(S_t,N_t),
\]

where \(S_t\) contains variables relevant to future prediction and \(N_t\)
contains nuisance variation. Observations are generated by

\[
O_t=g(S_t,N_t,\varepsilon_t),
\]

and a future target is

\[
Y_t=h(S_{t+1:t+H},\zeta_t).
\]

The history available to the encoder is

\[
\mathcal H_t=(O_{t-L+1},\ldots,O_t).
\]

A token-preserving encoder produces a grid

\[
G_t=f_\theta(\mathcal H_t)\in\mathbb R^{K\times S\times d},
\]

and an explicitly chosen pooling or readout map \(P\) produces

\[
Z_t^{(P)}=P(G_t)\in\mathbb R^D.
\]

The distinction between \(G_t\) and \(Z_t^{(P)}\) is essential. A pooling can
remove information, cancel temporally inconsistent coordinates, or discard
role-specific variation before any downstream reader is fitted.

On real data, \(S_t\), the sufficient predictive statistic, and Bayes risk are
unknown. The best available quantities are therefore operational lower bounds
obtained from specified reader classes. In the proposed synthetic system these
objects will be known or numerically verifiable.

---

## 3. Four operational notions

### 3.1 Predictive content

For a loss \(\ell\), define the optimal risk obtainable from a representation
\(Z\) by

\[
\mathcal R^*(Z;Y)
=
\inf_{q\ \mathrm{measurable}}
\mathbb E[\ell(Y,q(Z))].
\]

Let

\[
\mathcal R_0^*(Y)
=
\inf_c \mathbb E[\ell(Y,c)]
\]

be the constant-predictor risk. A normalized notion of predictive content is

\[
C(Z;Y)
=
1-\frac{\mathcal R^*(Z;Y)}{\mathcal R_0^*(Y)}.
\]

When \(\mathcal R_0^*(Y)>0\), this is the population analogue of an optimal
nonlinear \(R^2\) under squared loss. It is invariant under bijective measurable
reparameterizations of \(Z\).

In practice, the unrestricted infimum is unavailable. For a reader class
\(\mathcal Q\), define

\[
\mathcal R_{\mathcal Q}^*(Z;Y)
=
\inf_{q\in\mathcal Q}
\mathbb E[\ell(Y,q(Z))].
\]

Linear and MLP probes estimate different class-restricted lower bounds on
content. They should not be described as the total information in the encoder,
especially when they operate on a pooled \(Z^{(P)}\) rather than the full grid
\(G\).

### 3.2 Resource-constrained accessibility

Accessibility describes how much predictive performance can be recovered
under an explicit resource budget. The budget includes:

- pooling \(P\);
- retained dimension \(m\);
- labelled sample size \(n\);
- reader class \(\mathcal Q\);
- regularization \(\lambda\);
- fitting algorithm and optimization budget.

We therefore regard accessibility as a family

\[
A_{P,\mathcal Q,\lambda}(m,n),
\]

not as a scalar intrinsic to an encoder. For the current PCA ladder, \(P\) is
fixed, \(U_m\) contains the top \(m\) eigenvectors of
\(\Sigma_Z=\operatorname{Cov}(Z)\), and the measured quantity is

\[
A_{\mathrm{PCA}}(m)
=
\frac{R^2(U_m^\top Z;Y)}{R^2(Z;Y)}.
\]

This ratio is reported only when the full-rank denominator is materially
positive. Otherwise raw risks or raw \(R^2\) values must be used; a normalized
ladder with a near-zero denominator is not interpretable.

A numerical random-subspace null uses Haar-distributed orthonormal
\(Q_m\in\mathbb R^{D\times m}\) in the same reader protocol:

\[
A_{\mathrm{null}}(m)
=
\mathbb E_{Q_m}
\left[
\frac{R^2(Q_m^\top Z;Y)}{R^2(Z;Y)}
\right].
\]

Accessibility is deliberately not invariant to arbitrary invertible changes of
coordinates. This is not a defect in the definition: finite samples,
regularization, dimension reduction, and optimization all operate in a chosen
metric and coordinate system. The empirical object of interest is precisely
the consequence of that choice.

### 3.3 Variance–task alignment

PCA orders directions by marginal variance, not predictive relevance. For
square-integrable \(Z\) and \(Y\), a target-aware population operator in
whitened coordinates is

\[
M_{\mathrm{pred}}
=
\Sigma_Z^{-1/2}
\Sigma_{ZY}
\Sigma_Y^{-1}
\Sigma_{YZ}
\Sigma_Z^{-1/2},
\]

with regularized inverses when covariance matrices are singular or poorly
conditioned. Its leading eigenspaces identify directions carrying large
whitened linear association with the target.

Let \(U_m\) be the leading covariance eigenspace and \(V_r\) the leading
predictive eigenspace of \(M_{\mathrm{pred}}\). Both are compared in whitened
coefficient geometry: a coefficient subspace \(B\) in the original
representation maps to
\(\operatorname{span}(\Sigma_Z^{1/2}B)\). For a PCA eigenspace this mapping
preserves its span. A simple predictive-energy statistic is then

\[
\operatorname{Align}(m,r)
=
\frac{1}{r}\lVert U_m^\top V_r\rVert_F^2,
\]

supplemented by principal angles and covariance-weighted predictive energy.
This is a proposed population formulation. The existing Euclidean
coefficient-angle diagnostic is not yet equivalent to predictive \(R^2\), and
the covariance-weighted version remains to be executed.

### 3.4 Decoder co-adaptation

Strict decoder co-adaptation should compare a decoder trained jointly with an
encoder against a fresh decoder of the same class, fitted after freezing the
encoder with matched data and optimization:

\[
\Delta_{\mathrm{coadapt}}
=
\mathcal R_{\mathrm{fresh}}
-
\mathcal R_{\mathrm{native}}.
\]

This quantity is distinct from content, accessibility, and target
generalization. It can be expanded through sample-size curves, decoder swaps,
and cross-seed alignment.

The present held-out experiment is not yet a direct estimate of
\(\Delta_{\mathrm{coadapt}}\). It asks whether supervised accessibility
survives on targets not optimized during training. Its result is evidence of
task-specific alignment and possible co-adaptation, not a causal native/fresh
identification.

---

## 4. Non-identifiability of internal geometry

Consider a decoded predictor

\[
\widehat Y=WZ.
\]

For any invertible \(T\),

\[
Z'=TZ,
\qquad
W'=WT^{-1}
\]

gives

\[
W'Z'=WZ.
\]

Thus a predictive function can admit multiple factorizations with identical
outputs and risk but different internal covariance, conditioning, coefficient
norms, and finite-resource accessibility. This linear indeterminacy is related
to established identifiability and representation-similarity questions
[[1](#references), [2](#references)]; it is not presented here as a new
theorem.

The research question begins after this observation. A concrete learning
objective, optimizer, normalization scheme, architecture, and regularizer may
break the indeterminacy and select a particular representative. Euclidean
latent-prediction losses are not invariant to every non-orthogonal
transformation, for example. We ask which geometry is selected, whether that
selection is systematic across objectives, and what its operational
consequences are.

### Proposition 1 — content can be fixed while top-PCA accessibility changes

Let \(S,N,\epsilon\) be centered, mutually independent scalar random variables
with positive finite variances. Define

\[
Y=S+\epsilon,
\qquad
Z_{a,b}=(aS,bN),
\]

for \(a,b\neq0\).

Then:

1. the optimal full linear predictor from \(Z_{a,b}\) is
   \(\widehat Y=Z_1/a=S\), with risk
   \(\operatorname{Var}(\epsilon)\), independent of \(a\) and \(b\);
2. if
   \[
   b^2\operatorname{Var}(N)
   >
   a^2\operatorname{Var}(S),
   \]
   the first principal direction contains only \(N\), is independent of \(Y\),
   and has zero explained variance;
3. if the inequality is reversed, the first principal direction contains
   \(S\) and reaches the full linear predictive performance.

**Proof.** Independence makes the covariance of \(Z_{a,b}\) diagonal with
entries \(a^2\operatorname{Var}(S)\) and
\(b^2\operatorname{Var}(N)\). PCA therefore selects the coordinate with the
larger entry. The first coordinate determines \(S\), while the second is
independent of \(Y\). The stated risks follow immediately. \(\square\)

This example separates information-preserving geometry from accessibility. It
does not explain why a learned encoder chooses particular \(a\) and \(b\).
That selection problem is the proposed mechanistic target.

With finite \(n\), observation noise, and ridge regularization, rescaling also
changes conditioning and estimation error even without a hard PCA truncation.
Consequently, sample-efficiency curves are a more general endpoint than the
variance-ordered ladder alone.

---

## 5. Evidence from the real system

The empirical study uses three encoder seeds per training objective. The
post-consolidation extraction contains \(100{,}000\) train and \(50{,}000\)
validation endpoints with zero stock-day overlap. A previously detected
endpoint-indexing error was corrected before the current extraction; hashes,
split membership, endpoint order, and artifact integrity are checked
fail-closed. Nonlinear probes use five reader seeds per encoder seed and early
stopping.

### 5.1 Variance–task anti-alignment

At \(m/D=1/32\):

| representation | top-PCA fraction | random-subspace null | full linear \(R^2\) |
|---|---:|---:|---:|
| horizon-JEPA, last | 0.0050 | 0.0563 | 0.2111 |
| masked-JEPA, last | 0.0022 | 0.0488 | 0.1006 |
| supervised, last | 0.8971 | 0.6118 | 0.3756 |

![Normalized accessibility for the final concatenated token
readout](../results/phase2/figures/01_predictive_mass.png)

*Figure 1. Fraction of full-rank directional \(R^2\) recovered as a function
of variance-ordered rank. Solid lines show the observed PCA ladder, dotted
lines the numerical random-subspace null, and the black dashed line the
analytic \(m/D\) reference. The analytic diagonal is not an appropriate null
for the anisotropic empirical representations.*

Both supervised and horizon-JEPA have low effective dimension. Their difference
is therefore better described as target alignment than as
isotropy-versus-anisotropy:

| representation | participation ratio | effective rank |
|---|---:|---:|
| horizon-JEPA, last | 7.91 | 11.75 |
| masked-JEPA, last | 45.31 | 88.69 |
| supervised, last | 7.05 | 14.46 |

Masked-JEPA is more diffuse yet less predictive, providing an internal control
against the claim that higher effective rank is automatically more useful.

### 5.2 Pooling dependence and the matched token-role diagnostic

The historical post-P0 nonlinear reader—not the later Phase-III-R reader—gives:

| pooling | horizon-JEPA | masked-JEPA | supervised |
|---|---:|---:|---:|
| final token concatenation | 0.3191 | 0.1501 | 0.3881 |
| temporal mean, token concatenation | 0.0494 | 0.0008 | 0.3917 |

The horizon/supervised ratio is \(82.23\%\) at the final position but
\(12.62\%\) after temporal averaging. This does not prove that the horizon
representation is intrinsically temporally incoherent: it establishes that the
chosen pooling is not information-neutral.

After selecting the endpoint readout, a Hadamard rotation across the four token
roles defines one fixed all-ones component and its three-dimensional zero-sum
complement in role space. With 128 channels per role, these are respectively
128- and 384-dimensional feature blocks. The historical linear diagnostic
reported:

\[
R^2_{\mathrm{common}}\approx0.041,
\qquad
R^2_{\mathrm{contrasts}}\approx0.205,
\qquad
R^2_{\mathrm{full}}\approx0.211.
\]

These values establish an operational fact: the fixed 128-dimensional
all-ones role projection retains little of horizon-JEPA's full-readout linear
performance, whereas it retains nearly all supervised performance in the
corresponding historical diagnostic. They do **not** establish that the
zero-sum role complement is intrinsically special. Its 384 dimensions are
three times the dimension of the all-ones block, and the independently fitted
out-of-sample values are not additive:

\[
R^2_{\mathrm{common}}+R^2_{\mathrm{contrasts}}
\ne R^2_{\mathrm{full}}.
\]

Orthogonality of the Hadamard basis on the role index does not imply
statistical independence of the projected feature blocks.

The preregistered T2 diagnostic subsequently compared the observed blocks with
100 deterministic Haar rotations in the four-dimensional role space, using
the same 128/384-dimensional common/complement split and the same historical
OLS reader. The historical 1,188 per-target cells were first reproduced with
maximum absolute error (1.79\times10^{-12}). For horizon-JEPA
`last_concat512`, the common-block lower-tail probabilities were
(0.099,0.059,0.099) across encoder seeds and the complement upper-tail
probabilities were (0.129,0.129,0.099). Thus neither preregistered condition
held in all three seeds. The same joint decision failed for `meanK_concatS`,
masked-JEPA and supervised. Shuffled-target block means lay between
(-0.0042) and (-0.0011), as expected for a null reader.

Consequently the structured role-null explanation is **not rejected**. This
does not erase the cross-encoder fixed-projection difference; it removes the
stronger claim that the Hadamard all-ones axis or its zero-sum complement is a
privileged relational mechanism. Temporal averaging, role projection and PCA
anti-alignment remain three distinct empirical operators. The mathematical
simulator should encode the spectral and temporal phenomena directly and may
include the all-ones projection only as an operational observation, not as a
special mechanism.

### 5.3 Held-out targets give a mixed result

For depth and imbalance, the supervised low-rank advantage largely disappears
at full rank or with a nonlinear fresh reader. Horizon-JEPA slightly exceeds
supervised with the final-position MLP:

| held-out target | horizon-JEPA | supervised |
|---|---:|---:|
| imbalance | 0.201 | 0.193 |
| depth | 0.174 | 0.162 |

Timing behaves differently. With temporal-mean MLP, supervised retains
\(R^2=0.570\) against \(0.478\) for horizon-JEPA. However, linear screening
gives

\[
R^2(\text{timing}\mid14\ \text{training targets})=0.1817,
\]

which is not negligible and does not exclude nonlinear dependence. The correct
conclusion is mixed:

- depth and imbalance support strong task-specific alignment;
- timing suggests a more general organizational difference;
- strict decoder co-adaptation remains unmeasured.

---

## 6. Falsifiable hypotheses

### H1 — objective-dependent allocation

Under a matched process, encoder family, and optimization budget, different
training objectives systematically allocate predictive content differently
across covariance directions, token relations, and time.

**Evidence against H1:** the differences disappear across controlled synthetic
regimes or are explained entirely by unequal predictive content.

### H2 — content/accessibility separation

There exist regimes in which two learned representations have comparable
Bayes-relative content but materially different
\(A_{P,\mathcal Q,\lambda}(m,n)\), conditioning, or finite-sample reader risk.

**Evidence against H2:** accessibility becomes equivalent whenever content,
pooling, reader class, and sample size are controlled.

### H3 — geometric intervention

An explicit geometric regularizer can improve finite-sample accessibility
without adding downstream supervision and without materially reducing
Bayes-relative content.

The minimal causal comparison is:

| arm | predictive loss | teacher/predictor | regularizer |
|---|---|---|---|
| H0 | matched horizon loss | fixed | none |
| H-VIC | matched horizon loss | fixed | variance + covariance |
| H-SIG | matched horizon loss | fixed | SIGReg |

**Evidence against H3:** the regularizer changes spectrum or Gaussianity but
does not improve reader sample efficiency, or improves accessibility only by
discarding predictive content.

### H4 — co-adaptation

A measurable part of native-decoder performance arises from joint selection of
an encoder coordinate system and decoder, beyond the content accessible to a
fresh matched decoder.

**Evidence against H4:** native and fresh sample-efficiency curves coincide
after matching data, model class, regularization, and optimization.

These hypotheses are deliberately separable. In particular, increased
isotropy does not by itself imply better target alignment, and a successful
intervention need not make a representation resemble the supervised
covariance spectrum.

---

## 7. Proposed controlled replication and open questions

### 7.1 Minimal stochastic family

The proposed starting point is

\[
S_{t+1}=F_SS_t+\eta_t,
\qquad
N_{t+1}=F_NN_t+\xi_t,
\]

\[
O_t=g(A_SS_t+A_NN_t+\varepsilon_t),
\qquad
Y_t=B S_{t+h}+\zeta_t.
\]

Here \(S_t\) is predictive state and \(N_t\) is nuisance state. The model should
make the following quantities known or verifiable:

- sufficient predictive statistic;
- Bayes predictor and Bayes risk;
- predictive and nuisance subspaces;
- observability;
- population covariance;
- signal-to-noise ratio.

The controlled parameters are predictive/nuisance variance, persistence,
mixing, observation noise, horizon, latent dimension, and eventually the
nonlinearity \(g\).

The first analysis should be linear and, where possible, analytic. Its purpose
is not merely to hand-code a high-variance nuisance variable—the toy example
already does that—but to determine which solution is selected by alternative
learning objectives and regularizers.

### 7.2 Experimental sequence

1. **Population analysis.** Derive sufficient state, covariance, prediction
   operators, and equivalent encoder–decoder factorizations.
2. **Linear learned encoders.** Compare supervised, horizon-predictive, and
   masked/reconstruction objectives under controlled constraints.
3. **Small neural replication.** Use a matched encoder family and preregistered
   content/accessibility endpoints.
4. **Geometric intervention.** Compare H0, H-VIC, and H-SIG first in the
   simulator.
5. **Real-system transfer.** Run the same matched intervention on the LOB only
   if the controlled experiment isolates an interpretable mechanism.

A full LeJEPA implementation should not be the first intervention because it
can simultaneously change the target encoder, predictor, stop-gradient
structure, and regularizer. VICReg and SIGReg are useful first because they
permit a narrower causal comparison
[[5](#references), [6](#references)]. Prior work also indicates that predictive
joint-embedding objectives may preferentially retain slow features under
specific distractor dynamics [[4](#references)], making controlled persistence
of \(S_t\) and \(N_t\) an important axis.

### 7.3 Optional topological extension

If the linear mechanism is established, \(S_t\) may be replaced by a diffusion
on \(S^1\), a torus, or a switching manifold with known topology. Persistent
homology or cohomology would then test preservation of a ground-truth cycle or
circular coordinate [[8](#references)]. Applying topological summaries
post-hoc to the real LOB point cloud is outside the initial scope.

### 7.4 Questions for mathematical supervision

1. Is the proposed separation between invariant content and
   coordinate-dependent accessibility formulated at the correct level?
2. Which accessibility functional admits useful finite-sample bounds without
   becoming tied to one arbitrary reader?
3. Can the solution-selection problem be characterized analytically for a
   minimal linear predictive objective?
4. Which equivalence group is appropriate for supervised prediction,
   latent-space prediction, and normalized embeddings?
5. Does a topology-controlled extension add a genuine independent contribution,
   or should the thesis remain spectral and geometric?

The immediate objective is to answer these questions before committing to a
large simulator or new real-data training campaign.

---

## Preliminary positioning

Predictive state representations establish that state can be represented
through predictions of future observable events rather than through a
reconstructed hidden generative state [[3](#references)]. The present question
is different but adjacent: among representations that support prediction, what
selects their internal geometry and determines finite-resource readout?

Linear identifiability results characterize broad families of learned
representations up to linear indeterminacy [[1](#references)], while work on
representation similarity explains why unrestricted invariance to invertible
linear transforms can itself erase meaningful distinctions
[[2](#references)]. Our intended use of this literature is not to claim the
indeterminacy as new, but to study the operational quantities that are not
invariant to it.

VICReg explicitly controls per-coordinate variance and off-diagonal covariance
[[5](#references)]. LeJEPA proposes an isotropic-Gaussian target distribution
and SIGReg motivated by downstream prediction risk [[6](#references)]. These
works make geometric regularization a natural intervention, but they do not by
themselves establish that isotropy is optimal for a fixed family of
domain-specific targets, temporally overlapping observations, or
token-preserving dynamical representations.

This positioning is preliminary. A systematic review of identifiability,
linear self-supervised learning, predictive representations, finite-sample
probing, and representation topology is required before stating a novelty
claim.

---

## References

1. Roeder, G., Metz, L., and Kingma, D. P. [On Linear Identifiability of
   Learned Representations](https://arxiv.org/abs/2007.00810), 2020.
2. Kornblith, S., Norouzi, M., Lee, H., and Hinton, G. [Similarity of Neural
   Network Representations Revisited](https://arxiv.org/abs/1905.00414), ICML
   2019.
3. Littman, M. L., Sutton, R. S., and Singh, S. [Predictive Representations of
   State](https://papers.nips.cc/paper_files/paper/2001/hash/1e4d36177d71bbb3558e43af9577d70e-Abstract.html),
   NeurIPS 2001.
4. Sobal, V. et al. [Joint Embedding Predictive Architectures Focus on Slow
   Features](https://arxiv.org/abs/2211.10831), 2022.
5. Bardes, A., Ponce, J., and LeCun, Y. [VICReg:
   Variance-Invariance-Covariance Regularization for Self-Supervised
   Learning](https://arxiv.org/abs/2105.04906), ICLR 2022.
6. Balestriero, R. and LeCun, Y. [LeJEPA: Provable and Scalable
   Self-Supervised Learning Without the
   Heuristics](https://arxiv.org/abs/2511.08544), 2025.
7. Assran, M. et al. [Self-Supervised Learning from Images with a
   Joint-Embedding Predictive
   Architecture](https://arxiv.org/abs/2301.08243), CVPR 2023.
8. de Silva, V. and Vejdemo-Johansson, M. [Persistent Cohomology and Circular
   Coordinates](https://arxiv.org/abs/0905.4887), 2009.

---

## Local provenance

The empirical values in this note are documented in:

- `../history/STATO_TESI_POST_CONSOLIDAMENTO_20260728.md`;
- `../history/CONSOLIDATION_20260728.md`;
- `validation/readouts_v2_20260728/analysis_manifest.json`;
- `validation/readouts_v2_20260728/analysis_consolidation_20260728/`.

Canonical source state:

- P0 fix: `17c5ffd`;
- post-P0 consolidation: `6a94bd5`.
